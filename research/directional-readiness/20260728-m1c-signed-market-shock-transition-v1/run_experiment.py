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
STATE_PATH: Final[Path] = Path(
    "/Users/michaelsalerno/Documents/Codex/"
    "2026-07-23-you-are-working-in-the-github-5/data/cache/"
    "minimal-intraday-iv-excess-holdout-v0/frozen_state_surface.parquet"
)
HISTORICAL_OPTIONS_PATH: Final[Path] = Path(
    "/Users/michaelsalerno/Documents/Codex/"
    "2026-07-23-you-are-working-in-the-github-3/research/cross-market-context/"
    "20260723-daily-stock-front-options-context-v01/artifacts/primary/"
    "front_options_dimensions.parquet"
)
STRESS_OPTIONS_PATH: Final[Path] = (
    REPO_ROOT
    / "research"
    / "options-feasibility"
    / "20260724-minimal-intraday-iv-excess-holdout-v01"
    / "artifacts"
    / "primary"
    / "holdout_selected_option_pairs.parquet"
)
VTI_PATH: Final[Path] = Path(
    "/Users/michaelsalerno/StockerLocal/data/processed/source=eodhd/"
    "instrument_type=stock/symbol=VTI.US/timeframe=5m/data.parquet"
)
EXPECTED_HASHES: Final[dict[Path, str]] = {
    TAIL_EPISODES_PATH: "a843fedf4c5df712237fd374c5efb3ff8925b575c875395ea648d786b589d9a3",
    TAIL_CHECKPOINTS_PATH: "8dd0ef53d9c5493b70f600a28d6f77e8ffabd5e7b48a5378cf0bb4411382cb8f",
    STATE_PATH: "68b1cc53c1570d53054d685966eef96f533d8760368ebfc148766bb8f3a6bcc0",
    HISTORICAL_OPTIONS_PATH: (
        "4bc6fd0ce6972210949a5447fd06ca0ffaa258cb953d5e3447c1c07afab85b40"
    ),
    STRESS_OPTIONS_PATH: (
        "0b3f16cb06ae00df06dc34041a87d78b17478f1245810f54b5cc2d0f38d27e97"
    ),
}

sys.path.insert(0, str(REPO_ROOT / "packages" / "stocker_prospective" / "src"))
sys.path.insert(0, str(REPO_ROOT / "packages" / "stocker_research" / "src"))

from stocker_prospective.signed_market_shock_v1 import (  # noqa: E402
    FROZEN_SHOCK_CHECKPOINTS_V1,
    MARKET_SHOCK_PROXY_V1,
    CheckpointShockThresholdsV1,
    MarketShockBarV1,
    MarketShockStateResultV1,
    PreentryMarketWindowsV1,
    assert_unprotected_sessions_v1,
    calculate_preentry_windows_v1,
    calculate_stock_shock_response_v1,
    classify_market_shock_state_v1,
    frozen_material_move_v1,
    partition_material_endpoint_v1,
)
from stocker_research.m1c_low_movement_v0 import iv_expected_absolute  # noqa: E402
from stocker_research.m1c_signed_market_shock_transition_v1 import (  # noqa: E402
    MINIMUM_PREDICTOR_SUPPORT_V1,
    assign_response_quintile_v1,
    freeze_checkpoint_thresholds_v1,
    freeze_response_quintiles_v1,
)

DEVELOPMENT_START: Final[str] = "2024-01-01"
DEVELOPMENT_END: Final[str] = "2024-12-31"
ASSESSMENT_START: Final[str] = "2025-01-01"
ASSESSMENT_END: Final[str] = "2025-08-22"
STRESS_START: Final[str] = "2025-09-01"
STRESS_END: Final[str] = "2025-12-31"
PROTECTED_START: Final[str] = "2026-01-01"
BOOTSTRAP_DRAWS: Final[int] = 5000
SESSION_BOOTSTRAP_SEED: Final[int] = 2026072803
EVENT_BOOTSTRAP_SEED: Final[int] = 2026072804
NULL_DRAWS: Final[int] = 1000
NULL_SEED: Final[int] = 2026072805
IDENTITY: Final[list[str]] = ["stock", "session", "checkpoint"]
ONSET_STATES: Final[tuple[str, str]] = (
    "NEGATIVE_SHOCK_ONSET",
    "POSITIVE_SHOCK_ONSET",
)
STATE_ORDER: Final[tuple[str, ...]] = (
    "NEGATIVE_SHOCK_ONSET",
    "POSITIVE_SHOCK_ONSET",
    "ONGOING_NEGATIVE_SHOCK",
    "ONGOING_POSITIVE_SHOCK",
    "ELEVATED_RANGE_NONDIRECTIONAL",
    "NORMAL_OTHER",
    "UNKNOWN_INCOMPLETE",
)
RUN_COMMAND: Final[str] = (
    "rtk uv run python research/directional-readiness/"
    "20260728-m1c-signed-market-shock-transition-v1/run_experiment.py"
)
FOCUSED_TEST_COMMAND: Final[str] = (
    "rtk uv run pytest tests/test_m1c_signed_market_shock_transition_v1.py "
    "tests/test_m1c_signed_market_shock_transition_v1_artifacts.py "
    "tests/test_m1c_signed_market_shock_transition_v1_recorder.py -q"
)
TARGETED_RUFF_COMMAND: Final[str] = (
    "rtk uv run ruff check "
    "packages/stocker_prospective/src/stocker_prospective/config.py "
    "packages/stocker_prospective/src/stocker_prospective/frozen_live_application.py "
    "packages/stocker_prospective/src/stocker_prospective/live_recorder.py "
    "packages/stocker_prospective/src/stocker_prospective/recorder_repository.py "
    "packages/stocker_prospective/src/stocker_prospective/recorder_v0.py "
    "packages/stocker_prospective/src/stocker_prospective/signed_market_shock_v1.py "
    "packages/stocker_research/src/stocker_research/"
    "m1c_signed_market_shock_transition_v1.py "
    "research/directional-readiness/"
    "20260728-m1c-signed-market-shock-transition-v1/run_experiment.py "
    "tests/test_m1c_signed_market_shock_transition_v1.py "
    "tests/test_m1c_signed_market_shock_transition_v1_artifacts.py "
    "tests/test_m1c_signed_market_shock_transition_v1_recorder.py"
)
TARGETED_MYPY_COMMAND: Final[str] = (
    "rtk uv run mypy "
    "packages/stocker_prospective/src/stocker_prospective/config.py "
    "packages/stocker_prospective/src/stocker_prospective/frozen_live_application.py "
    "packages/stocker_prospective/src/stocker_prospective/live_recorder.py "
    "packages/stocker_prospective/src/stocker_prospective/recorder_repository.py "
    "packages/stocker_prospective/src/stocker_prospective/recorder_v0.py "
    "packages/stocker_prospective/src/stocker_prospective/signed_market_shock_v1.py "
    "packages/stocker_research/src/stocker_research/"
    "m1c_signed_market_shock_transition_v1.py"
)


class ExperimentBlocked(RuntimeError):
    """A scientific or operational prerequisite failed closed."""


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_,)):
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
        raise ExperimentBlocked("blocked_market_proxy_data")
    prior = cast(
        dict[str, Any],
        json.loads(TAIL_PROVENANCE_PATH.read_text(encoding="utf-8")),
    )
    protected = cast(dict[str, Any], prior["protected_data_confirmation"])
    execution = cast(dict[str, Any], prior["execution_confirmation"])
    causes = cast(dict[str, Any], prior["causality_confirmation"])
    if any(
        (
            bool(protected["protected_data_opened"]),
            bool(protected["protected_outcomes_calculated"]),
            bool(protected["protected_outcomes_displayed"]),
            bool(protected["protected_outcomes_inspected"]),
            bool(execution["broker_access"]),
            bool(execution["order_routing_enabled"]),
            bool(execution["orders_submitted"]),
            bool(causes["m1c_refit"]),
            bool(causes["a1_refit"]),
            bool(causes["fresh_episode_definition_changed"]),
            bool(causes["archived_signed_pressure_used"]),
            bool(causes["archived_tension_used"]),
        )
    ):
        raise ExperimentBlocked("inherited frozen-system provenance violates V1")
    records.append(
        {
            "path": str(TAIL_PROVENANCE_PATH),
            "sha256": _sha256_file(TAIL_PROVENANCE_PATH),
            "bytes": TAIL_PROVENANCE_PATH.stat().st_size,
            "hash_scope": "whole_file_known_to_end_by_2025-12-31",
        }
    )
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
]:
    episodes = _opened_session_read(TAIL_EPISODES_PATH)
    checkpoints = _opened_session_read(TAIL_CHECKPOINTS_PATH)
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
        "volume",
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
    if not bars["bar_is_complete"].astype(bool).all():
        raise ExperimentBlocked("stock bar source contains incomplete bars")
    if bars.duplicated(["stock", "session", "bar_ordinal"]).any():
        raise ExperimentBlocked("stock bar identities are not unique")

    option_columns = ["symbol", "session", "previous_close_underlying_price", "atm_iv"]
    historical = pd.read_parquet(
        HISTORICAL_OPTIONS_PATH,
        columns=option_columns,
        filters=[
            ("session", ">=", DEVELOPMENT_START),
            ("session", "<=", ASSESSMENT_END),
        ],
    )
    stress = pd.read_parquet(
        STRESS_OPTIONS_PATH,
        columns=[*option_columns, "pair_available"],
        filters=[
            ("session", ">=", STRESS_START),
            ("session", "<=", STRESS_END),
        ],
    )
    stress = stress.loc[stress["pair_available"].astype(bool), option_columns]
    options = pd.concat([historical, stress], ignore_index=True).rename(
        columns={"symbol": "stock"}
    )
    assert_unprotected_sessions_v1(options["session"])
    if options.duplicated(["stock", "session"]).any():
        raise ExperimentBlocked("prior-close IV context is not unique")

    vti = pd.read_parquet(
        VTI_PATH,
        columns=["source", "symbol", "timeframe", "timestamp", "open", "high", "low", "close"],
        filters=[
            ("timestamp", ">=", datetime(2024, 1, 1, tzinfo=UTC)),
            ("timestamp", "<", datetime(2026, 1, 1, tzinfo=UTC)),
        ],
    )
    timestamps = pd.to_datetime(vti["timestamp"], utc=True, errors="raise")
    if timestamps.ge(pd.Timestamp(PROTECTED_START, tz=UTC)).any():
        raise ExperimentBlocked("protected VTI row admitted by bounded reader")
    return episodes, checkpoints, bars, options, vti


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
    frame["symbol"] = MARKET_SHOCK_PROXY_V1
    frame["partition"] = frame["session"].map(_partition)
    frame = frame.loc[frame["partition"].notna()].copy()
    if frame.duplicated(["session", "bar_ordinal"]).any():
        raise ExperimentBlocked("canonical VTI bars are not unique")
    prices = frame[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
    valid = (
        np.isfinite(prices.to_numpy(float)).all(axis=1)
        & prices.gt(0.0).all(axis=1)
        & prices["high"].ge(prices[["open", "close", "low"]].max(axis=1))
        & prices["low"].le(prices[["open", "close", "high"]].min(axis=1))
    )
    frame["finalised"] = valid.to_numpy(bool)
    frame["source_ohlc_valid_v1"] = valid.to_numpy(bool)
    frame = frame.sort_values(["session", "bar_ordinal"], kind="mergesort").reset_index(drop=True)
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


def _market_bar_groups(frame: pd.DataFrame) -> dict[str, tuple[MarketShockBarV1, ...]]:
    groups: dict[str, tuple[MarketShockBarV1, ...]] = {}
    for session, group in frame.groupby("session", sort=False):
        groups[str(session)] = tuple(
            MarketShockBarV1(
                symbol=MARKET_SHOCK_PROXY_V1,
                session=date.fromisoformat(str(row.session)),
                bar_ordinal=int(row.bar_ordinal),
                bar_start_timestamp=pd.Timestamp(row.bar_start_timestamp).to_pydatetime(),
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


def _market_schedule() -> pd.DataFrame:
    import pandas_market_calendars as market_calendars

    calendar = market_calendars.get_calendar("NYSE")
    schedule = calendar.schedule(
        start_date=DEVELOPMENT_START,
        end_date=STRESS_END,
    ).reset_index(names="session")
    schedule["session"] = pd.to_datetime(schedule["session"]).dt.strftime("%Y-%m-%d")
    schedule["partition"] = schedule["session"].map(_partition)
    return schedule.loc[schedule["partition"].notna()].reset_index(drop=True)


def _build_market_windows(vti_bars: pd.DataFrame) -> pd.DataFrame:
    groups = _market_bar_groups(vti_bars)
    records: list[dict[str, Any]] = []
    for row in _market_schedule().itertuples(index=False):
        session = str(row.session)
        market_open = pd.Timestamp(row.market_open).to_pydatetime()
        for checkpoint in FROZEN_SHOCK_CHECKPOINTS_V1:
            signal = market_open + timedelta(minutes=5 * checkpoint)
            measurement = calculate_preentry_windows_v1(
                market_proxy=MARKET_SHOCK_PROXY_V1,
                session=date.fromisoformat(session),
                checkpoint=checkpoint,
                signal_timestamp=signal,
                completed_bars=groups.get(session, ()),
            )
            records.append(
                {
                    "session": session,
                    "partition": str(row.partition),
                    **measurement.model_dump(mode="python"),
                }
            )
    return pd.DataFrame(records)


def _apply_market_states(
    market_windows: pd.DataFrame,
    thresholds: Mapping[int, CheckpointShockThresholdsV1],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in market_windows.itertuples(index=False):
        raw = cast(Any, row)
        windows = PreentryMarketWindowsV1(
            market_proxy_v1=str(raw.market_proxy_v1),
            session=(
                raw.session
                if isinstance(raw.session, date)
                else date.fromisoformat(str(raw.session))
            ),
            checkpoint=int(raw.checkpoint),
            signal_timestamp=pd.Timestamp(raw.signal_timestamp).to_pydatetime(),
            w0_bar_ordinals_v1=tuple(raw.w0_bar_ordinals_v1),
            w1_bar_ordinals_v1=tuple(raw.w1_bar_ordinals_v1),
            market_return_w0_v1=raw.market_return_w0_v1,
            market_range_w0_v1=raw.market_range_w0_v1,
            market_return_w1_v1=raw.market_return_w1_v1,
            market_range_w1_v1=raw.market_range_w1_v1,
            maximum_market_timestamp_v1=(
                None
                if pd.isna(raw.maximum_market_timestamp_v1)
                else pd.Timestamp(raw.maximum_market_timestamp_v1).to_pydatetime()
            ),
            complete_v1=bool(raw.complete_v1),
            missing_reasons_v1=tuple(raw.missing_reasons_v1),
        )
        state = classify_market_shock_state_v1(
            windows=windows,
            thresholds=thresholds.get(int(raw.checkpoint)),
        )
        threshold = thresholds[int(raw.checkpoint)]
        records.append(
            {
                "session": str(raw.session),
                "checkpoint": int(raw.checkpoint),
                "partition": str(raw.partition),
                "market_proxy_v1": windows.market_proxy_v1,
                "signal_timestamp": windows.signal_timestamp,
                "w0_bar_ordinals_v1": windows.w0_bar_ordinals_v1,
                "w1_bar_ordinals_v1": windows.w1_bar_ordinals_v1,
                "market_return_w0_v1": windows.market_return_w0_v1,
                "market_range_w0_v1": windows.market_range_w0_v1,
                "market_return_w1_v1": windows.market_return_w1_v1,
                "market_range_w1_v1": windows.market_range_w1_v1,
                "maximum_market_timestamp_v1": windows.maximum_market_timestamp_v1,
                "market_window_complete_v1": windows.complete_v1,
                "market_window_missing_reasons_v1": windows.missing_reasons_v1,
                "market_shock_state_v1": state.market_shock_state_v1,
                "market_shock_event_id_v1": state.market_shock_event_id_v1,
                "shock_sign_v1": state.shock_sign_v1,
                "market_shock_complete_v1": state.complete_v1,
                "market_shock_missing_reasons_v1": state.missing_reasons_v1,
                **{
                    f"threshold_{key}": value
                    for key, value in threshold.model_dump(mode="python").items()
                    if key != "checkpoint"
                },
            }
        )
    return pd.DataFrame(records)


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
    duplicate_consistency = stock_bars.groupby(
        ["session", "bar_ordinal"], sort=False
    )[["vti__bar_log_return", "vti__bar_range_pct"]].nunique(dropna=False)
    if (duplicate_consistency > 1).any().any():
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
    available_after_bar = (
        pd.to_datetime(comparison["benchmark_available_timestamp"], utc=True)
        > pd.to_datetime(comparison["bar_complete_timestamp"], utc=True)
    )
    maximum_return = float(return_difference.max()) if len(return_difference) else math.nan
    maximum_range = float(range_difference.max()) if len(range_difference) else math.nan
    if (
        not math.isfinite(maximum_return)
        or maximum_return > 1e-12
        or not math.isfinite(maximum_range)
        or maximum_range > 1e-12
        or available_after_bar.any()
    ):
        raise ExperimentBlocked("canonical VTI causal alignment regression failed")
    return {
        "rows_compared": int(valid.sum()),
        "maximum_absolute_bar_return_difference": maximum_return,
        "maximum_absolute_bar_range_difference": maximum_range,
        "benchmark_available_after_stock_bar_count": int(available_after_bar.sum()),
        "passed": True,
    }


def _option_context(options: pd.DataFrame) -> pd.DataFrame:
    context = options.copy()
    iv = pd.to_numeric(context["atm_iv"], errors="coerce")
    context["threshold_15m"] = iv.map(
        lambda value: (
            iv_expected_absolute(float(value), 15)
            if math.isfinite(float(value)) and float(value) > 0.0
            else math.nan
        )
    )
    context["implied_movement_15m_price"] = (
        pd.to_numeric(context["previous_close_underlying_price"], errors="coerce")
        * context["threshold_15m"]
    )
    return context


def _attach_endpoint_outcomes(
    panel: pd.DataFrame,
    bars: pd.DataFrame,
) -> pd.DataFrame:
    output = panel.copy()
    entry = bars[["stock", "session", "bar_ordinal", "open"]].rename(
        columns={"bar_ordinal": "checkpoint", "open": "_entry_open_15m"}
    )
    terminal = bars[["stock", "session", "bar_ordinal", "close"]].copy()
    terminal["checkpoint"] = terminal["bar_ordinal"].astype(int) - 2
    terminal = terminal.rename(columns={"close": "_terminal_close_15m"}).drop(
        columns="bar_ordinal"
    )
    output = output.merge(
        entry,
        on=IDENTITY,
        how="left",
        validate="one_to_one",
    ).merge(
        terminal,
        on=IDENTITY,
        how="left",
        validate="one_to_one",
    )
    entry_values = pd.to_numeric(output["_entry_open_15m"], errors="coerce").to_numpy(float)
    terminal_values = pd.to_numeric(
        output["_terminal_close_15m"], errors="coerce"
    ).to_numpy(float)
    thresholds = pd.to_numeric(output["threshold_15m"], errors="coerce").to_numpy(float)
    valid = (
        np.isfinite(entry_values)
        & (entry_values > 0.0)
        & np.isfinite(terminal_values)
        & (terminal_values > 0.0)
        & np.isfinite(thresholds)
        & (thresholds > 0.0)
    )
    signed = np.full(len(output), np.nan, dtype=float)
    signed[valid] = np.log(terminal_values[valid] / entry_values[valid])
    output["future_signed_return_15m"] = signed
    output["primary_outcome_complete_v1"] = valid
    output["primary_outcome_state_v1"] = "UNKNOWN_INCOMPLETE"
    output.loc[valid, "primary_outcome_state_v1"] = [
        partition_material_endpoint_v1(
            signed_return=float(observed),
            threshold_15m=float(threshold),
        )
        for observed, threshold in zip(signed[valid], thresholds[valid], strict=True)
    ]
    output["future_absolute_movement_15m_v1"] = np.abs(signed)
    output["future_iv_residual_15m_v1"] = np.abs(signed) - thresholds
    output["future_exceed_iv_15m_v1"] = (
        output["primary_outcome_state_v1"].isin(["MATERIAL_UP", "MATERIAL_DOWN"])
    ).where(output["primary_outcome_complete_v1"])
    comparable = (
        output["primary_outcome_complete_v1"]
        & output.get("future_15m_absolute_movement_v1", pd.Series(index=output.index)).notna()
    )
    if comparable.any():
        absolute_difference = np.abs(
            pd.to_numeric(
                output.loc[comparable, "future_15m_absolute_movement_v1"],
                errors="coerce",
            )
            - output.loc[comparable, "future_absolute_movement_15m_v1"]
        )
        residual_difference = np.abs(
            pd.to_numeric(
                output.loc[comparable, "future_15m_iv_residual_v1"],
                errors="coerce",
            )
            - output.loc[comparable, "future_iv_residual_15m_v1"]
        )
        archived_exceed = output.loc[comparable, "future_15m_exceed_iv_v1"].astype(bool)
        reconstructed_exceed = output.loc[
            comparable, "future_exceed_iv_15m_v1"
        ].astype(bool)
        if (
            float(absolute_difference.max()) > 1e-12
            or float(residual_difference.max()) > 1e-12
            or not archived_exceed.equals(reconstructed_exceed)
        ):
            raise ExperimentBlocked("strict frozen M1C endpoint reconstruction drifted")
    equivalence = output.loc[valid, "primary_outcome_state_v1"].isin(
        ["MATERIAL_UP", "MATERIAL_DOWN"]
    )
    strict = pd.Series(
        [
            frozen_material_move_v1(
                signed_return=float(observed),
                threshold_15m=float(threshold),
            )
            for observed, threshold in zip(signed[valid], thresholds[valid], strict=True)
        ],
        index=output.index[valid],
    )
    if not equivalence.equals(strict):
        raise ExperimentBlocked("strict target partition equivalence failed")
    return output


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
    for (stock, session), group in selected.groupby(["stock", "session"], sort=False):
        cache[(str(stock), str(session))] = tuple(
            MarketShockBarV1(
                symbol=str(stock),
                session=date.fromisoformat(str(session)),
                bar_ordinal=int(row.bar_ordinal),
                bar_start_timestamp=pd.Timestamp(row.bar_start_timestamp).to_pydatetime(),
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
            for row in group.sort_values("bar_ordinal", kind="mergesort").itertuples(
                index=False
            )
        )
    return cache


def _attach_stock_responses(
    panel: pd.DataFrame,
    bars: pd.DataFrame,
) -> pd.DataFrame:
    output = panel.copy()
    cache = _stock_bar_cache(bars, output)
    records: list[dict[str, Any]] = []
    for raw in output.itertuples(index=False):
        row = cast(Any, raw)
        session = date.fromisoformat(str(row.session))
        market_state = MarketShockStateResultV1(
            market_shock_state_v1=str(row.market_shock_state_v1),
            market_shock_event_id_v1=(
                None
                if pd.isna(row.market_shock_event_id_v1)
                else str(row.market_shock_event_id_v1)
            ),
            shock_sign_v1=(
                None if pd.isna(row.shock_sign_v1) else int(row.shock_sign_v1)
            ),
            complete_v1=bool(row.market_shock_complete_v1),
            missing_reasons_v1=tuple(row.market_shock_missing_reasons_v1),
        )
        result = calculate_stock_shock_response_v1(
            symbol=str(row.stock),
            session=session,
            checkpoint=int(row.checkpoint),
            signal_timestamp=pd.Timestamp(row.signal_timestamp).to_pydatetime(),
            completed_stock_bars=cache.get((str(row.stock), str(row.session)), ()),
            market_return_w0_v1=(
                None
                if pd.isna(row.market_return_w0_v1)
                else float(row.market_return_w0_v1)
            ),
            market_shock_state_v1=market_state,
            threshold_15m=(
                None if pd.isna(row.threshold_15m) else float(row.threshold_15m)
            ),
        )
        stock_bars = {
            bar.bar_ordinal: bar
            for bar in cache.get((str(row.stock), str(row.session)), ())
        }
        previous = stock_bars.get(int(row.checkpoint) - 2)
        latest = stock_bars.get(int(row.checkpoint) - 1)
        recent_5m = (
            math.log(latest.close / previous.close)
            if previous is not None
            and latest is not None
            and previous.close > 0.0
            and latest.close > 0.0
            else math.nan
        )
        records.append(
            {
                **result.model_dump(mode="python"),
                "recent_stock_return_5m_v1": recent_5m,
            }
        )
    response = pd.DataFrame(records, index=output.index).rename(
        columns={
            "complete_v1": "shock_response_complete_v1",
            "missing_reasons_v1": "shock_response_missing_reasons_v1",
        }
    )
    for column in response:
        output[column] = response[column]
    signed = pd.to_numeric(output["future_signed_return_15m"], errors="coerce")
    sign = pd.to_numeric(output["shock_sign_v1"], errors="coerce")
    output["continuation_aligned_return_v1"] = sign * signed
    output["resistance_aligned_return_v1"] = -sign * signed
    output["followed_shock_v1"] = pd.Series(pd.NA, index=output.index, dtype="Int64")
    material = output["primary_outcome_state_v1"].isin(
        ["MATERIAL_UP", "MATERIAL_DOWN"]
    ) & sign.notna()
    material_direction = np.where(
        output.loc[material, "primary_outcome_state_v1"].eq("MATERIAL_UP"),
        1,
        -1,
    )
    output.loc[material, "followed_shock_v1"] = (
        material_direction == sign.loc[material].to_numpy(float)
    ).astype(int)
    output["continuation_action_v1"] = "ABSTAIN"
    output.loc[
        output["shock_response_class_v1"].eq("AMPLIFYING") & sign.eq(1),
        "continuation_action_v1",
    ] = "CALL"
    output.loc[
        output["shock_response_class_v1"].eq("AMPLIFYING") & sign.eq(-1),
        "continuation_action_v1",
    ] = "PUT"
    output["resistance_action_v1"] = "ABSTAIN"
    output.loc[
        output["shock_response_class_v1"].eq("RESISTING") & sign.eq(1),
        "resistance_action_v1",
    ] = "PUT"
    output.loc[
        output["shock_response_class_v1"].eq("RESISTING") & sign.eq(-1),
        "resistance_action_v1",
    ] = "CALL"
    output["month_v1"] = output["session"].astype(str).str[:7]
    return output


def _prepare_panel(
    source: pd.DataFrame,
    *,
    population: str,
    bars: pd.DataFrame,
    options: pd.DataFrame,
    market_states: pd.DataFrame,
    attach_responses: bool,
) -> pd.DataFrame:
    if source.duplicated(IDENTITY).any():
        raise ExperimentBlocked(f"{population} identities are not unique")
    output = source.copy()
    assert_unprotected_sessions_v1(output["session"])
    output = output.merge(
        _option_context(options),
        on=["stock", "session"],
        how="left",
        validate="many_to_one",
    )
    market_columns = [
        column for column in market_states.columns if column != "partition"
    ]
    market_panel = market_states[market_columns].rename(
        columns={"signal_timestamp": "market_signal_timestamp_v1"}
    )
    output = output.merge(
        market_panel,
        on=["session", "checkpoint"],
        how="left",
        validate="many_to_one",
    )
    source_signal = pd.to_datetime(output["signal_timestamp"], utc=True, errors="coerce")
    market_signal = pd.to_datetime(
        output["market_signal_timestamp_v1"], utc=True, errors="coerce"
    )
    feature_available = pd.to_datetime(
        output["feature_available_timestamp_utc"],
        utc=True,
        errors="coerce",
    )
    if (
        not source_signal.equals(market_signal)
        or not source_signal.equals(feature_available)
    ):
        raise ExperimentBlocked("market and M1C signal timestamps differ")
    output = _attach_endpoint_outcomes(output, bars)
    output["population_v1"] = population
    if attach_responses:
        output = _attach_stock_responses(output, bars)
    return output


def _frozen_regression(
    source: pd.DataFrame,
    experiment: pd.DataFrame,
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
    comparison = source[[*IDENTITY, *fields]].merge(
        experiment[[*IDENTITY, *fields]],
        on=IDENTITY,
        suffixes=("_source", "_experiment"),
        validate="one_to_one",
    )
    passed = len(comparison) == len(source)
    for field in fields:
        left = comparison[f"{field}_source"]
        right = comparison[f"{field}_experiment"]
        if pd.api.types.is_numeric_dtype(left):
            passed = passed and bool(
                np.allclose(
                    pd.to_numeric(left, errors="coerce"),
                    pd.to_numeric(right, errors="coerce"),
                    rtol=0.0,
                    atol=0.0,
                    equal_nan=True,
                )
            )
        else:
            passed = passed and left.equals(right)
    if not passed:
        raise ExperimentBlocked("frozen M1C/Tail/A1/fresh-episode regression failed")
    return {
        "rows_compared": int(len(comparison)),
        "fields": fields,
        "tolerance": 0.0,
        "passed": True,
    }


def _distribution(values: pd.Series) -> str:
    clean = values.dropna().astype(str)
    counts = clean.value_counts(dropna=False).sort_index()
    total = int(counts.sum())
    payload = {
        key: {
            "count": int(value),
            "rate": float(value / total) if total else None,
        }
        for key, value in counts.items()
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _maximum_share(values: pd.Series) -> float:
    clean = values.dropna()
    if clean.empty:
        return math.nan
    return float(clean.value_counts(normalize=True).max())


def _outcome_metrics(
    frame: pd.DataFrame,
    *,
    period: str,
    group_type: str,
    group_value: str,
) -> dict[str, Any]:
    state = frame["primary_outcome_state_v1"]
    rows = int(len(frame))
    up = int(state.eq("MATERIAL_UP").sum())
    down = int(state.eq("MATERIAL_DOWN").sum())
    no_move = int(state.eq("NO_MATERIAL_MOVE").sum())
    signed = pd.to_numeric(frame["future_signed_return_15m"], errors="coerce")
    absolute = signed.abs()
    residual = pd.to_numeric(frame["future_iv_residual_15m_v1"], errors="coerce")
    exceed = frame["future_exceed_iv_15m_v1"].astype("boolean")
    remaining = pd.to_numeric(
        frame.get("post_share_of_local_range_v1", pd.Series(index=frame.index)),
        errors="coerce",
    )
    return {
        "period": period,
        "group_type": group_type,
        "group_value": group_value,
        "episode_count": rows,
        "unique_shock_event_count": int(
            frame["market_shock_event_id_v1"].dropna().nunique()
        ),
        "session_count": int(frame["session"].nunique()),
        "stock_count": int(frame["stock"].nunique()),
        "material_up_count": up,
        "material_up_rate": float(up / rows) if rows else math.nan,
        "material_down_count": down,
        "material_down_rate": float(down / rows) if rows else math.nan,
        "no_material_move_count": no_move,
        "no_material_move_rate": float(no_move / rows) if rows else math.nan,
        "mean_signed_15m_return": float(signed.mean()) if rows else math.nan,
        "median_signed_15m_return": float(signed.median()) if rows else math.nan,
        "mean_absolute_15m_movement": float(absolute.mean()) if rows else math.nan,
        "median_absolute_15m_movement": float(absolute.median()) if rows else math.nan,
        "mean_iv_residual": float(residual.mean()) if rows else math.nan,
        "median_iv_residual": float(residual.median()) if rows else math.nan,
        "exceed_iv_rate": float(exceed.mean()) if rows else math.nan,
        "mean_post_entry_remaining_range_share": (
            float(remaining.mean()) if remaining.notna().any() else math.nan
        ),
        "tail_phase_composition_json": _distribution(frame["m1c_tail_phase_v1"]),
        "checkpoint_distribution_json": _distribution(frame["checkpoint"]),
        "time_of_day_distribution_json": _distribution(frame["time_of_day_v1"]),
        "stock_distribution_json": _distribution(frame["stock"]),
        "session_distribution_json": _distribution(frame["session"]),
        "shock_event_distribution_json": _distribution(
            frame["market_shock_event_id_v1"]
        ),
        "maximum_stock_share": _maximum_share(frame["stock"]),
        "maximum_session_share": _maximum_share(frame["session"]),
        "maximum_shock_event_share": _maximum_share(
            frame["market_shock_event_id_v1"]
        ),
        "maximum_checkpoint_share": _maximum_share(frame["checkpoint"]),
    }


def _required_outcome_table(frame: pd.DataFrame, *, period: str) -> pd.DataFrame:
    working = frame.loc[
        frame["partition"].eq(period)
        & frame["market_shock_state_v1"].isin(ONSET_STATES)
    ].copy()
    specifications: list[tuple[str, str, pd.Series]] = [
        ("population", "ALL_SIGNED_SHOCK_ONSETS", pd.Series(True, index=working.index)),
        (
            "shock_sign",
            "NEGATIVE_SHOCK_ONSET",
            working["market_shock_state_v1"].eq("NEGATIVE_SHOCK_ONSET"),
        ),
        (
            "shock_sign",
            "POSITIVE_SHOCK_ONSET",
            working["market_shock_state_v1"].eq("POSITIVE_SHOCK_ONSET"),
        ),
    ]
    for response_class in (
        "AMPLIFYING",
        "RESISTING",
        "NEUTRAL_EXACT",
        "NOT_SHOCK_ONSET",
        "UNKNOWN_INCOMPLETE",
    ):
        specifications.append(
            (
                "response_class",
                response_class,
                working["shock_response_class_v1"].eq(response_class),
            )
        )
    for subtype in (
        "RESISTING_BUT_STILL_WITH_SHOCK",
        "ABSOLUTELY_OPPOSING_SHOCK",
    ):
        specifications.append(
            (
                "resisting_descriptive_subtype",
                subtype,
                working["resisting_subtype_v1"].eq(subtype),
            )
        )
    for state in ONSET_STATES:
        for response_class in (
            "AMPLIFYING",
            "RESISTING",
            "NEUTRAL_EXACT",
            "NOT_SHOCK_ONSET",
            "UNKNOWN_INCOMPLETE",
        ):
            specifications.append(
                (
                    "shock_sign_x_response_class",
                    f"{state}|{response_class}",
                    working["market_shock_state_v1"].eq(state)
                    & working["shock_response_class_v1"].eq(response_class),
                )
            )
    return pd.DataFrame(
        [
            _outcome_metrics(
                working.loc[mask].copy(),
                period=period,
                group_type=group_type,
                group_value=group_value,
            )
            for group_type, group_value, mask in specifications
        ]
    )


def _unique_shock_events(
    market_states: pd.DataFrame,
    episodes: pd.DataFrame,
) -> pd.DataFrame:
    events = market_states.loc[
        market_states["market_shock_state_v1"].isin(ONSET_STATES)
    ][
        [
            "partition",
            "session",
            "checkpoint",
            "market_proxy_v1",
            "market_shock_state_v1",
            "market_shock_event_id_v1",
            "shock_sign_v1",
            "market_return_w0_v1",
            "market_range_w0_v1",
            "market_return_w1_v1",
            "market_range_w1_v1",
        ]
    ].copy()
    if events["market_shock_event_id_v1"].duplicated().any():
        raise ExperimentBlocked("market shock event identifiers are not unique")
    episode_counts = (
        episodes.loc[episodes["market_shock_state_v1"].isin(ONSET_STATES)]
        .groupby("market_shock_event_id_v1", sort=False)
        .agg(
            fresh_high_m1c_episode_count=("stock", "size"),
            represented_stock_count=("stock", "nunique"),
            amplifying_episode_count=(
                "shock_response_class_v1",
                lambda values: int(values.eq("AMPLIFYING").sum()),
            ),
            resisting_episode_count=(
                "shock_response_class_v1",
                lambda values: int(values.eq("RESISTING").sum()),
            ),
            neutral_exact_episode_count=(
                "shock_response_class_v1",
                lambda values: int(values.eq("NEUTRAL_EXACT").sum()),
            ),
        )
        .reset_index()
    )
    events = events.merge(
        episode_counts,
        on="market_shock_event_id_v1",
        how="left",
        validate="one_to_one",
    )
    count_columns = [
        "fresh_high_m1c_episode_count",
        "represented_stock_count",
        "amplifying_episode_count",
        "resisting_episode_count",
        "neutral_exact_episode_count",
    ]
    events[count_columns] = events[count_columns].fillna(0).astype(int)
    return events


def _event_accounting(
    market_states: pd.DataFrame,
    episodes: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for period in ("development", "assessment", "stress"):
        market = market_states.loc[market_states["partition"].eq(period)]
        primary_checkpoints = market.loc[market["checkpoint"].ne(6)]
        complete_by_session = primary_checkpoints.groupby("session", sort=False)[
            "market_window_complete_v1"
        ].all()
        episode_period = episodes.loc[episodes["partition"].eq(period)]
        event_period = events.loc[events["partition"].eq(period)]
        incomplete_episode = (
            episode_period["market_shock_state_v1"].eq("UNKNOWN_INCOMPLETE")
            | episode_period["shock_response_class_v1"].eq("UNKNOWN_INCOMPLETE")
        )
        events_with_fresh = event_period.loc[
            event_period["fresh_high_m1c_episode_count"].gt(0)
        ]
        record: dict[str, Any] = {
            "period": period,
            "sessions_with_complete_market_data": int(complete_by_session.sum()),
            "scheduled_sessions": int(market["session"].nunique()),
            "unique_negative_shock_onsets": int(
                market["market_shock_state_v1"].eq("NEGATIVE_SHOCK_ONSET").sum()
            ),
            "unique_positive_shock_onsets": int(
                market["market_shock_state_v1"].eq("POSITIVE_SHOCK_ONSET").sum()
            ),
            "ongoing_shocks": int(
                market["market_shock_state_v1"].isin(
                    ["ONGOING_NEGATIVE_SHOCK", "ONGOING_POSITIVE_SHOCK"]
                ).sum()
            ),
            "elevated_range_nondirectional_states": int(
                market["market_shock_state_v1"]
                .eq("ELEVATED_RANGE_NONDIRECTIONAL")
                .sum()
            ),
            "unique_market_shock_event_ids": int(
                event_period["market_shock_event_id_v1"].nunique()
            ),
            "unique_market_shock_event_ids_with_fresh_high_m1c": int(
                events_with_fresh["market_shock_event_id_v1"].nunique()
            ),
            "mean_stocks_per_shock_event": (
                float(event_period["represented_stock_count"].mean())
                if len(event_period)
                else math.nan
            ),
            "mean_stocks_per_shock_event_with_fresh_high_m1c": (
                float(events_with_fresh["represented_stock_count"].mean())
                if len(events_with_fresh)
                else math.nan
            ),
            "median_stocks_per_shock_event": (
                float(event_period["represented_stock_count"].median())
                if len(event_period)
                else math.nan
            ),
            "incomplete_fresh_episodes": int(incomplete_episode.sum()),
        }
        for state in STATE_ORDER:
            record[f"fresh_high_m1c_episodes__{state}"] = int(
                episode_period["market_shock_state_v1"].eq(state).sum()
            )
        records.append(record)
    return pd.DataFrame(records)


def _prediction_from_action(actions: pd.Series) -> pd.Series:
    output = pd.Series(0, index=actions.index, dtype=int)
    output.loc[actions.astype(str).eq("CALL")] = 1
    output.loc[actions.astype(str).eq("PUT")] = -1
    return output


def _sign_prediction(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    output = pd.Series(0, index=values.index, dtype=int)
    output.loc[numeric.gt(0.0)] = 1
    output.loc[numeric.lt(0.0)] = -1
    return output


def _policy_metrics(
    frame: pd.DataFrame,
    *,
    prediction: pd.Series,
    period: str,
    arm: str,
    policy: str,
    scope: str,
) -> dict[str, Any]:
    predicted = prediction.reindex(frame.index).fillna(0).astype(int)
    acted = predicted.ne(0)
    working = frame.loc[acted].copy()
    signs = predicted.loc[acted]
    signed = pd.to_numeric(working["future_signed_return_15m"], errors="coerce")
    aligned = signs.to_numpy(float) * signed.to_numpy(float)
    outcome = working["primary_outcome_state_v1"]
    material = outcome.isin(["MATERIAL_UP", "MATERIAL_DOWN"])
    actual_sign = pd.Series(
        np.where(outcome.eq("MATERIAL_UP"), 1, np.where(outcome.eq("MATERIAL_DOWN"), -1, 0)),
        index=working.index,
    )
    correct = signs.eq(actual_sign) & material
    opposing = signs.eq(-actual_sign) & material
    aligned_series = pd.Series(aligned, index=working.index)
    session_returns = aligned_series.groupby(working["session"], sort=False).mean()
    month_returns = aligned_series.groupby(working["month_v1"], sort=False).mean()
    return {
        "period": period,
        "arm": arm,
        "policy": policy,
        "evaluation_scope": scope,
        "eligible_episode_count": int(len(frame)),
        "acted_episode_count": int(acted.sum()),
        "abstention_count": int((~acted).sum()),
        "unique_shock_event_count": int(
            working["market_shock_event_id_v1"].dropna().nunique()
        ),
        "session_count": int(working["session"].nunique()),
        "stock_count": int(working["stock"].nunique()),
        "call_count": int(signs.eq(1).sum()),
        "put_count": int(signs.eq(-1).sum()),
        "material_up_count": int(outcome.eq("MATERIAL_UP").sum()),
        "material_down_count": int(outcome.eq("MATERIAL_DOWN").sum()),
        "no_material_move_count": int(outcome.eq("NO_MATERIAL_MOVE").sum()),
        "material_direction_accuracy": (
            float(correct.loc[material].mean()) if material.any() else math.nan
        ),
        "accuracy_counting_no_move_as_failure": (
            float(correct.mean()) if len(correct) else math.nan
        ),
        "material_following_prediction_rate": (
            float(correct.sum() / material.sum()) if material.any() else math.nan
        ),
        "material_opposing_prediction_rate": (
            float(opposing.sum() / material.sum()) if material.any() else math.nan
        ),
        "material_following_minus_opposing_rate": (
            float((correct.sum() - opposing.sum()) / material.sum())
            if material.any()
            else math.nan
        ),
        "mean_aligned_return": (
            float(aligned_series.mean()) if len(aligned_series) else math.nan
        ),
        "median_aligned_return": (
            float(aligned_series.median()) if len(aligned_series) else math.nan
        ),
        "positive_session_rate": (
            float(session_returns.gt(0.0).mean()) if len(session_returns) else math.nan
        ),
        "positive_month_rate": (
            float(month_returns.gt(0.0).mean()) if len(month_returns) else math.nan
        ),
        "maximum_stock_share": _maximum_share(working["stock"]),
        "maximum_session_share": _maximum_share(working["session"]),
        "maximum_shock_event_share": _maximum_share(
            working["market_shock_event_id_v1"]
        ),
        "maximum_checkpoint_share": _maximum_share(working["checkpoint"]),
    }


def _baseline_tables(frame: pd.DataFrame, *, period: str) -> pd.DataFrame:
    onset = frame.loc[
        frame["partition"].eq(period)
        & frame["market_shock_state_v1"].isin(ONSET_STATES)
        & frame["primary_outcome_complete_v1"].astype(bool)
    ].copy()
    sign = pd.to_numeric(onset["shock_sign_v1"], errors="coerce").fillna(0).astype(int)
    predictions = {
        "follow_market_shock": sign,
        "oppose_market_shock": -sign,
        "recent_stock_momentum_5m": _sign_prediction(onset["recent_stock_return_5m_v1"]),
        "trailing_stock_momentum_15m": _sign_prediction(onset["stock_return_w0_v1"]),
        "frozen_A1": _prediction_from_action(onset["A1_action_v1"]),
        "always_CALL": pd.Series(1, index=onset.index),
        "always_PUT": pd.Series(-1, index=onset.index),
    }
    records: list[dict[str, Any]] = []
    scopes = [
        ("all_signed_shock_onsets", pd.Series(True, index=onset.index), "all"),
        (
            "continuation_amplifying_acted",
            onset["shock_response_class_v1"].eq("AMPLIFYING"),
            "continuation",
        ),
        (
            "resistance_resisting_acted",
            onset["shock_response_class_v1"].eq("RESISTING"),
            "resistance",
        ),
    ]
    for scope, mask, arm in scopes:
        subset = onset.loc[mask].copy()
        for policy, prediction in predictions.items():
            records.append(
                _policy_metrics(
                    subset,
                    prediction=prediction,
                    period=period,
                    arm=arm,
                    policy=policy,
                    scope=scope,
                )
            )
        arm_prediction = (
            _prediction_from_action(subset["continuation_action_v1"])
            if arm == "continuation"
            else (
                _prediction_from_action(subset["resistance_action_v1"])
                if arm == "resistance"
                else pd.Series(0, index=subset.index)
            )
        )
        if arm in {"continuation", "resistance"}:
            records.append(
                _policy_metrics(
                    subset,
                    prediction=arm_prediction,
                    period=period,
                    arm=arm,
                    policy=f"{arm}_v1",
                    scope=scope,
                )
            )
    identity_columns = {"period", "arm", "policy", "evaluation_scope"}
    blocked = {
        column: None for column in records[0] if column not in identity_columns
    }
    for scope, _, arm in scopes:
        records.append(
            {
                "period": period,
                "arm": arm,
                "policy": "existing_frozen_D2",
                "evaluation_scope": scope,
                **blocked,
                "status": "blocked_contaminated_or_unreproducible_lineage",
            }
        )
    return pd.DataFrame(records)


def _safe_auc(target: pd.Series, score: pd.Series) -> float:
    valid = target.notna() & score.notna()
    observed = target.loc[valid].astype(int)
    values = pd.to_numeric(score.loc[valid], errors="coerce")
    finite = np.isfinite(values.to_numpy(float))
    observed = observed.loc[finite]
    values = values.loc[finite]
    if len(observed) < 2 or observed.nunique() != 2:
        return math.nan
    return float(roc_auc_score(observed, values))


def _safe_spearman(left: pd.Series, right: pd.Series) -> tuple[float, float]:
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
    statistic = spearmanr(left_values, right_values)
    return float(statistic.statistic), float(statistic.pvalue)


def _ranking_tables(
    frame: pd.DataFrame,
    *,
    period: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    onset = frame.loc[
        frame["partition"].eq(period)
        & frame["market_shock_state_v1"].isin(ONSET_STATES)
        & frame["shock_response_complete_v1"].astype(bool)
        & frame["primary_outcome_complete_v1"].astype(bool)
    ].copy()
    material = onset.loc[onset["followed_shock_v1"].notna()].copy()
    correlation, correlation_p = _safe_spearman(
        onset["shock_relative_response_v1"],
        onset["continuation_aligned_return_v1"],
    )
    summary = pd.DataFrame(
        [
            {
                "period": period,
                "eligible_shock_onset_episode_count": int(len(onset)),
                "material_mover_count": int(len(material)),
                "unique_shock_event_count": int(
                    onset["market_shock_event_id_v1"].nunique()
                ),
                "session_count": int(onset["session"].nunique()),
                "roc_auc_followed_shock_among_material_movers": _safe_auc(
                    material["followed_shock_v1"],
                    material["shock_relative_response_v1"],
                ),
                "spearman_response_vs_continuation_aligned_return": correlation,
                "spearman_two_sided_p_value": correlation_p,
            }
        ]
    )
    quintile_records: list[dict[str, Any]] = []
    for quintile in ("Q1", "Q2", "Q3", "Q4", "Q5", "UNKNOWN_INCOMPLETE"):
        group = onset.loc[onset["response_quintile_v1"].eq(quintile)].copy()
        group_material = group.loc[group["followed_shock_v1"].notna()]
        quintile_records.append(
            {
                "period": period,
                "response_quintile_v1": quintile,
                "episode_count": int(len(group)),
                "material_mover_count": int(len(group_material)),
                "unique_shock_event_count": int(
                    group["market_shock_event_id_v1"].nunique()
                ),
                "session_count": int(group["session"].nunique()),
                "followed_shock_rate_among_material_movers": (
                    float(group_material["followed_shock_v1"].astype(float).mean())
                    if len(group_material)
                    else math.nan
                ),
                "mean_continuation_aligned_return": (
                    float(group["continuation_aligned_return_v1"].mean())
                    if len(group)
                    else math.nan
                ),
                "no_material_move_rate": (
                    float(group["primary_outcome_state_v1"].eq("NO_MATERIAL_MOVE").mean())
                    if len(group)
                    else math.nan
                ),
            }
        )
    return summary, pd.DataFrame(quintile_records)


def _bootstrap_statistic(sample: pd.DataFrame) -> dict[str, float]:
    amplifying = sample.loc[
        sample["shock_response_class_v1"].eq("AMPLIFYING")
    ]
    resisting = sample.loc[sample["shock_response_class_v1"].eq("RESISTING")]
    material = sample.loc[sample["followed_shock_v1"].notna()]
    return {
        "continuation_mean_aligned_return": (
            float(amplifying["continuation_aligned_return_v1"].mean())
            if len(amplifying)
            else math.nan
        ),
        "resistance_mean_aligned_return": (
            float(resisting["resistance_aligned_return_v1"].mean())
            if len(resisting)
            else math.nan
        ),
        "continuous_ranking_auc": _safe_auc(
            material["followed_shock_v1"],
            material["shock_relative_response_v1"],
        ),
    }


def _cluster_bootstrap(
    frame: pd.DataFrame,
    *,
    period: str,
    cluster_column: str,
    draws: int,
    seed: int,
) -> pd.DataFrame:
    onset = frame.loc[
        frame["partition"].eq(period)
        & frame["market_shock_state_v1"].isin(ONSET_STATES)
        & frame["primary_outcome_complete_v1"].astype(bool)
        & frame[cluster_column].notna()
    ].copy()
    clusters = onset[cluster_column].drop_duplicates().tolist()
    if not clusters:
        return pd.DataFrame()
    positions = {
        cluster: np.flatnonzero(onset[cluster_column].eq(cluster).to_numpy())
        for cluster in clusters
    }
    generator = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    for draw in range(draws):
        selected = generator.choice(clusters, size=len(clusters), replace=True)
        sampled_positions = np.concatenate([positions[value] for value in selected])
        sample = onset.iloc[sampled_positions]
        records.append(
            {
                "period": period,
                "cluster_type": cluster_column,
                "draw": draw,
                "seed": seed,
                **_bootstrap_statistic(sample),
            }
        )
    return pd.DataFrame(records)


def _bootstrap_summary(draws: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    if draws.empty:
        return pd.DataFrame()
    statistic_columns = (
        "continuation_mean_aligned_return",
        "resistance_mean_aligned_return",
        "continuous_ranking_auc",
    )
    for (period, cluster_type), group in draws.groupby(
        ["period", "cluster_type"],
        sort=False,
    ):
        for statistic in statistic_columns:
            values = pd.to_numeric(group[statistic], errors="coerce").dropna()
            records.append(
                {
                    "period": period,
                    "cluster_type": cluster_type,
                    "statistic": statistic,
                    "valid_draw_count": int(len(values)),
                    "lower_95": (
                        float(values.quantile(0.025)) if len(values) else math.nan
                    ),
                    "median": (
                        float(values.quantile(0.5)) if len(values) else math.nan
                    ),
                    "upper_95": (
                        float(values.quantile(0.975)) if len(values) else math.nan
                    ),
                }
            )
    return pd.DataFrame(records)


def _arm_frame(frame: pd.DataFrame, *, period: str, arm: str) -> pd.DataFrame:
    response_class = "AMPLIFYING" if arm == "continuation" else "RESISTING"
    return frame.loc[
        frame["partition"].eq(period)
        & frame["market_shock_state_v1"].isin(ONSET_STATES)
        & frame["shock_response_class_v1"].eq(response_class)
        & frame["primary_outcome_complete_v1"].astype(bool)
    ].copy()


def _support_result(frame: pd.DataFrame) -> dict[str, Any]:
    sign = pd.to_numeric(frame["shock_sign_v1"], errors="coerce")
    checks = {
        "minimum_stock_episodes": len(frame) >= 30,
        "minimum_sessions": frame["session"].nunique() >= 10,
        "minimum_market_shock_events": (
            frame["market_shock_event_id_v1"].nunique() >= 10
        ),
        "minimum_negative_actions": int(sign.eq(-1).sum()) >= 8,
        "minimum_positive_actions": int(sign.eq(1).sum()) >= 8,
        "minimum_stocks": frame["stock"].nunique() >= 8,
        "maximum_single_stock_share": _maximum_share(frame["stock"]) <= 0.25,
        "maximum_single_session_share": _maximum_share(frame["session"]) <= 0.20,
        "maximum_single_shock_event_share": (
            _maximum_share(frame["market_shock_event_id_v1"]) <= 0.20
        ),
        "maximum_single_checkpoint_share": (
            _maximum_share(frame["checkpoint"]) <= 0.40
        ),
    }
    episode_support = (
        checks["minimum_stock_episodes"]
        and checks["minimum_sessions"]
        and checks["minimum_negative_actions"]
        and checks["minimum_positive_actions"]
        and checks["minimum_stocks"]
        and checks["maximum_single_stock_share"]
        and checks["maximum_single_session_share"]
        and checks["maximum_single_checkpoint_share"]
    )
    if episode_support and not checks["minimum_market_shock_events"]:
        status = "blocked_insufficient_independent_shock_support"
    elif all(checks.values()):
        status = "support_pass"
    else:
        status = "blocked_insufficient_support"
    return {
        "support_status": status,
        "support_pass": bool(all(checks.values())),
        "episode_count": int(len(frame)),
        "session_count": int(frame["session"].nunique()),
        "market_shock_event_count": int(
            frame["market_shock_event_id_v1"].nunique()
        ),
        "negative_action_count": int(sign.eq(-1).sum()),
        "positive_action_count": int(sign.eq(1).sum()),
        "stock_count": int(frame["stock"].nunique()),
        "maximum_stock_share": _maximum_share(frame["stock"]),
        "maximum_session_share": _maximum_share(frame["session"]),
        "maximum_shock_event_share": _maximum_share(
            frame["market_shock_event_id_v1"]
        ),
        "maximum_checkpoint_share": _maximum_share(frame["checkpoint"]),
        "checks": checks,
    }


def _winsorised_mean(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return math.nan
    lower, upper = clean.quantile([0.01, 0.99])
    return float(clean.clip(lower=lower, upper=upper).mean())


def _arm_results(
    frame: pd.DataFrame,
    *,
    period: str,
    arm: str,
) -> pd.DataFrame:
    working = _arm_frame(frame, period=period, arm=arm)
    action_column = (
        "continuation_action_v1" if arm == "continuation" else "resistance_action_v1"
    )
    aligned_column = (
        "continuation_aligned_return_v1"
        if arm == "continuation"
        else "resistance_aligned_return_v1"
    )
    metrics = _policy_metrics(
        working,
        prediction=_prediction_from_action(working[action_column]),
        period=period,
        arm=arm,
        policy=f"{arm}_v1",
        scope=f"{arm}_acted",
    )
    support = _support_result(working)
    return pd.DataFrame(
        [
            {
                **metrics,
                "one_percent_winsorised_mean_aligned_return": _winsorised_mean(
                    working[aligned_column]
                ),
                "support_status": support["support_status"],
                "support_pass": support["support_pass"],
                "negative_shock_action_count": support["negative_action_count"],
                "positive_shock_action_count": support["positive_action_count"],
                "support_checks_json": json.dumps(
                    support["checks"],
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        ]
    )


def _checkpoint_stratified_mechanism_results(frame: pd.DataFrame) -> pd.DataFrame:
    """Report every frozen checkpoint for both preregistered mechanisms."""

    records: list[dict[str, Any]] = []
    for period in ("assessment", "stress"):
        for checkpoint in FROZEN_SHOCK_CHECKPOINTS_V1:
            for arm in ("continuation", "resistance"):
                working = _arm_frame(frame, period=period, arm=arm)
                working = working.loc[working["checkpoint"].eq(checkpoint)]
                action_column = (
                    "continuation_action_v1"
                    if arm == "continuation"
                    else "resistance_action_v1"
                )
                metrics = _policy_metrics(
                    working,
                    prediction=_prediction_from_action(working[action_column]),
                    period=period,
                    arm=arm,
                    policy=f"{arm}_v1",
                    scope=f"checkpoint_{checkpoint}",
                )
                support = _support_result(working)
                records.append(
                    {
                        **metrics,
                        "checkpoint": checkpoint,
                        "support_status": support["support_status"],
                        "support_checks_json": json.dumps(
                            support["checks"],
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    }
                )
    return pd.DataFrame(records)


def _leave_one_out(
    frame: pd.DataFrame,
    *,
    omitted_dimension: str,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for period in ("assessment", "stress"):
        for arm in ("continuation", "resistance"):
            working = _arm_frame(frame, period=period, arm=arm)
            aligned_column = (
                "continuation_aligned_return_v1"
                if arm == "continuation"
                else "resistance_aligned_return_v1"
            )
            for omitted_value in working[omitted_dimension].drop_duplicates():
                retained = working.loc[working[omitted_dimension].ne(omitted_value)]
                records.append(
                    {
                        "period": period,
                        "arm": arm,
                        "omitted_dimension": omitted_dimension,
                        "omitted_value": str(omitted_value),
                        "retained_episode_count": int(len(retained)),
                        "retained_session_count": int(retained["session"].nunique()),
                        "retained_shock_event_count": int(
                            retained["market_shock_event_id_v1"].nunique()
                        ),
                        "mean_aligned_return": (
                            float(retained[aligned_column].mean())
                            if len(retained)
                            else math.nan
                        ),
                    }
                )
    return pd.DataFrame(records)


def _concentration_report(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    dimensions = (
        ("stock", "stock"),
        ("month", "month_v1"),
        ("checkpoint", "checkpoint"),
        ("session", "session"),
        ("market_shock_event", "market_shock_event_id_v1"),
    )
    for period in ("assessment", "stress"):
        for arm in ("continuation", "resistance"):
            working = _arm_frame(frame, period=period, arm=arm)
            for dimension_name, column in dimensions:
                counts = working[column].value_counts(dropna=False)
                for value, count in counts.items():
                    records.append(
                        {
                            "period": period,
                            "arm": arm,
                            "dimension": dimension_name,
                            "value": str(value),
                            "episode_count": int(count),
                            "share": float(count / len(working)) if len(working) else math.nan,
                        }
                    )
    return pd.DataFrame(records)


def _null_statistics(sample: pd.DataFrame) -> dict[str, float]:
    sign = pd.to_numeric(sample["shock_sign_v1"], errors="coerce")
    outcome_sign = pd.to_numeric(sample["_null_outcome_sign"], errors="coerce")
    material = outcome_sign.ne(0)
    followed = pd.Series(
        np.where(material, outcome_sign.eq(sign).astype(int), np.nan),
        index=sample.index,
    )
    null_aligned = sign * pd.to_numeric(sample["_null_signed_return"], errors="coerce")
    amplifying = sample["shock_response_class_v1"].eq("AMPLIFYING")
    resisting = sample["shock_response_class_v1"].eq("RESISTING")
    continuation_correct = material & outcome_sign.eq(sign)
    resistance_correct = material & outcome_sign.eq(-sign)
    amplifying_material = amplifying & material
    resisting_material = resisting & material
    follow_amplifying = (
        float(followed.loc[amplifying_material].mean())
        if amplifying_material.any()
        else math.nan
    )
    follow_resisting = (
        float(followed.loc[resisting_material].mean())
        if resisting_material.any()
        else math.nan
    )
    return {
        "continuous_ranking_auc": _safe_auc(
            followed,
            sample["shock_relative_response_v1"],
        ),
        "continuation_mean_aligned_return": (
            float(null_aligned.loc[amplifying].mean())
            if amplifying.any()
            else math.nan
        ),
        "resistance_mean_aligned_return": (
            float((-null_aligned.loc[resisting]).mean())
            if resisting.any()
            else math.nan
        ),
        "continuation_material_direction_accuracy": (
            float(continuation_correct.loc[amplifying_material].mean())
            if amplifying_material.any()
            else math.nan
        ),
        "resistance_material_direction_accuracy": (
            float(resistance_correct.loc[resisting_material].mean())
            if resisting_material.any()
            else math.nan
        ),
        "amplifying_minus_resisting_follow_shock_rate": (
            follow_amplifying - follow_resisting
            if math.isfinite(follow_amplifying) and math.isfinite(follow_resisting)
            else math.nan
        ),
    }


def _primary_null(
    episodes: pd.DataFrame,
    donors: pd.DataFrame,
    *,
    period: str,
    draws: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    onset = episodes.loc[
        episodes["partition"].eq(period)
        & episodes["market_shock_state_v1"].isin(ONSET_STATES)
        & episodes["shock_response_complete_v1"].astype(bool)
        & episodes["primary_outcome_complete_v1"].astype(bool)
    ].copy()
    donor_pool = donors.loc[
        donors["partition"].eq(period)
        & donors["primary_outcome_complete_v1"].astype(bool)
    ].copy()
    pools: dict[tuple[str, int], pd.DataFrame] = {
        (str(stock), int(checkpoint)): group[
            ["session", "future_signed_return_15m", "primary_outcome_state_v1"]
        ].reset_index(drop=True)
        for (stock, checkpoint), group in donor_pool.groupby(
            ["stock", "checkpoint"],
            sort=False,
        )
    }
    candidates: list[pd.DataFrame] = []
    for row in onset.itertuples(index=False):
        pool = pools.get((str(row.stock), int(row.checkpoint)), pd.DataFrame())
        pool = pool.loc[pool["session"].astype(str).ne(str(row.session))].copy()
        if pool.empty:
            raise ExperimentBlocked(
                "primary null lacks a different-session donor for "
                f"{row.stock}/{row.checkpoint}/{period}"
            )
        candidates.append(pool)
    generator = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    for draw in range(draws):
        sample = onset.copy()
        signed_values: list[float] = []
        outcome_signs: list[int] = []
        for pool in candidates:
            chosen = pool.iloc[int(generator.integers(0, len(pool)))]
            signed_values.append(float(chosen["future_signed_return_15m"]))
            state = str(chosen["primary_outcome_state_v1"])
            outcome_signs.append(
                1 if state == "MATERIAL_UP" else (-1 if state == "MATERIAL_DOWN" else 0)
            )
        sample["_null_signed_return"] = signed_values
        sample["_null_outcome_sign"] = outcome_signs
        records.append(
            {
                "period": period,
                "draw": draw,
                "seed": seed,
                **_null_statistics(sample),
            }
        )
    audit = {
        "period": period,
        "eligible_predictor_rows": int(len(onset)),
        "donor_rows": int(len(donor_pool)),
        "grouping": ["stock", "checkpoint", "period"],
        "same_session_allowed": False,
        "outcome_completeness_preserved": True,
        "draws": draws,
        "seed": seed,
    }
    return pd.DataFrame(records), audit


def _temporal_placebo(frame: pd.DataFrame, *, period: str) -> dict[str, Any]:
    eligible = frame.loc[
        frame["partition"].eq(period)
        & frame["primary_outcome_complete_v1"].astype(bool)
    ].sort_values(["stock", "checkpoint", "session"], kind="mergesort")
    onset = eligible.loc[
        eligible["market_shock_state_v1"].isin(ONSET_STATES)
        & eligible["shock_response_complete_v1"].astype(bool)
    ].copy()
    outcome_groups = {
        (str(stock), int(checkpoint)): group.reset_index(drop=True)
        for (stock, checkpoint), group in eligible.groupby(
            ["stock", "checkpoint"],
            sort=False,
        )
    }
    records: list[dict[str, Any]] = []
    for row in onset.itertuples(index=False):
        candidates = outcome_groups[(str(row.stock), int(row.checkpoint))]
        candidates = candidates.loc[
            candidates["session"].astype(str).gt(str(row.session))
        ]
        if candidates.empty:
            continue
        outcome = candidates.iloc[0]
        record = row._asdict()
        record["_null_signed_return"] = float(outcome["future_signed_return_15m"])
        state = str(outcome["primary_outcome_state_v1"])
        record["_null_outcome_sign"] = (
            1 if state == "MATERIAL_UP" else (-1 if state == "MATERIAL_DOWN" else 0)
        )
        records.append(record)
    placebo = pd.DataFrame(records)
    statistics = (
        _null_statistics(placebo)
        if len(placebo)
        else {
            "continuous_ranking_auc": math.nan,
            "continuation_mean_aligned_return": math.nan,
            "resistance_mean_aligned_return": math.nan,
            "continuation_material_direction_accuracy": math.nan,
            "resistance_material_direction_accuracy": math.nan,
            "amplifying_minus_resisting_follow_shock_rate": math.nan,
        }
    )
    return {
        "period": period,
        "construction": (
            "next_eligible_fresh_high_m1c_outcome_same_stock_checkpoint_period"
        ),
        "paired_episode_count": int(len(placebo)),
        "cross_chronology_allowed": False,
        **statistics,
    }


def _one_sided_null_p(
    null_values: pd.Series,
    observed: float,
) -> float:
    values = pd.to_numeric(null_values, errors="coerce").dropna()
    if not math.isfinite(observed) or values.empty:
        return math.nan
    return float((1 + values.ge(observed).sum()) / (1 + len(values)))


def _holm_two(p_values: Mapping[str, float]) -> dict[str, float]:
    finite = [(key, value) for key, value in p_values.items() if math.isfinite(value)]
    if len(finite) != len(p_values):
        return {key: math.nan for key in p_values}
    ordered = sorted(finite, key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, (key, value) in enumerate(ordered):
        candidate = min(1.0, (count - rank) * value)
        running = max(running, candidate)
        adjusted[key] = running
    return adjusted


def _regime_metrics(
    frame: pd.DataFrame,
    *,
    period: str,
    regime: str,
    checkpoint: int | None,
) -> dict[str, Any]:
    if regime == "SIGNED_SHOCK_ONSET":
        mask = frame["market_shock_state_v1"].isin(ONSET_STATES)
        prediction = pd.to_numeric(frame["shock_sign_v1"], errors="coerce")
    else:
        mask = frame["market_shock_state_v1"].eq("NORMAL_OTHER")
        prediction = _sign_prediction(frame["market_return_w0_v1"])
    if checkpoint is not None:
        mask &= pd.to_numeric(frame["checkpoint"], errors="coerce").eq(checkpoint)
    working = frame.loc[
        frame["partition"].eq(period)
        & mask
        & frame["primary_outcome_complete_v1"].astype(bool)
    ].copy()
    signs = prediction.reindex(working.index).fillna(0).astype(int)
    acted = signs.ne(0)
    working = working.loc[acted]
    signs = signs.loc[acted]
    state = working["primary_outcome_state_v1"]
    actual = pd.Series(
        np.where(state.eq("MATERIAL_UP"), 1, np.where(state.eq("MATERIAL_DOWN"), -1, 0)),
        index=working.index,
    )
    material = actual.ne(0)
    correct = material & actual.eq(signs)
    signed = pd.to_numeric(working["future_signed_return_15m"], errors="coerce")
    return {
        "period": period,
        "regime": regime,
        "aggregation": "raw" if checkpoint is None else "checkpoint_stratum",
        "checkpoint": checkpoint,
        "episode_count": int(len(working)),
        "session_count": int(working["session"].nunique()),
        "checkpoint_count": int(working["checkpoint"].nunique()),
        "material_mover_count": int(material.sum()),
        "conditional_direction_accuracy_among_material_movers": (
            float(correct.loc[material].mean()) if material.any() else math.nan
        ),
        "accuracy_counting_no_move_as_failure": (
            float(correct.mean()) if len(correct) else math.nan
        ),
        "mean_market_aligned_15m_return": (
            float((signs * signed).mean()) if len(working) else math.nan
        ),
        "material_following_rate": (
            float(correct.sum() / material.sum()) if material.any() else math.nan
        ),
        "no_move_rate": (
            float(state.eq("NO_MATERIAL_MOVE").mean()) if len(state) else math.nan
        ),
        "iv_excess_rate": (
            float(working["future_exceed_iv_15m_v1"].astype(float).mean())
            if len(working)
            else math.nan
        ),
        "mean_absolute_15m_movement": (
            float(working["future_absolute_movement_15m_v1"].mean())
            if len(working)
            else math.nan
        ),
    }


def _normal_regime_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    metric_columns = (
        "conditional_direction_accuracy_among_material_movers",
        "accuracy_counting_no_move_as_failure",
        "mean_market_aligned_15m_return",
        "material_following_rate",
        "no_move_rate",
        "iv_excess_rate",
        "mean_absolute_15m_movement",
    )
    for period in ("assessment", "stress"):
        strata: dict[str, list[dict[str, Any]]] = {
            "SIGNED_SHOCK_ONSET": [],
            "NORMAL_OTHER": [],
        }
        for regime in strata:
            records.append(
                _regime_metrics(
                    frame,
                    period=period,
                    regime=regime,
                    checkpoint=None,
                )
            )
            for checkpoint in FROZEN_SHOCK_CHECKPOINTS_V1:
                row = _regime_metrics(
                    frame,
                    period=period,
                    regime=regime,
                    checkpoint=checkpoint,
                )
                strata[regime].append(row)
                records.append(row)
        common = [
            checkpoint
            for checkpoint in FROZEN_SHOCK_CHECKPOINTS_V1
            if all(
                next(
                    row["episode_count"]
                    for row in strata[regime]
                    if row["checkpoint"] == checkpoint
                )
                > 0
                for regime in strata
            )
        ]
        for regime, rows in strata.items():
            selected = [row for row in rows if row["checkpoint"] in common]
            records.append(
                {
                    "period": period,
                    "regime": regime,
                    "aggregation": "equal_weight_common_checkpoint_standardised",
                    "checkpoint": None,
                    "episode_count": int(sum(row["episode_count"] for row in selected)),
                    "session_count": math.nan,
                    "checkpoint_count": int(len(common)),
                    "material_mover_count": int(
                        sum(row["material_mover_count"] for row in selected)
                    ),
                    **{
                        column: (
                            float(
                                np.nanmean(
                                    np.asarray(
                                        [row[column] for row in selected],
                                        dtype=float,
                                    )
                                )
                            )
                            if selected
                            else math.nan
                        )
                        for column in metric_columns
                    },
                }
            )
    return pd.DataFrame(records)


def _tail_phase_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for period in ("assessment", "stress"):
        period_rows = frame.loc[frame["partition"].eq(period)]
        for state in STATE_ORDER:
            for phase in ("FIRST_ENTRY", "PERSISTENT", "RE_ENTRY"):
                group = period_rows.loc[
                    period_rows["market_shock_state_v1"].eq(state)
                    & period_rows["m1c_tail_phase_v1"].eq(phase)
                ].copy()
                support_status = (
                    "reported_descriptive"
                    if len(group) >= 30 and group["session"].nunique() >= 10
                    else "blocked_insufficient_support"
                )
                for policy_name, action_column in (
                    ("continuation_v1", "continuation_action_v1"),
                    ("resistance_v1", "resistance_action_v1"),
                    ("frozen_A1", "A1_action_v1"),
                ):
                    metrics = _policy_metrics(
                        group,
                        prediction=_prediction_from_action(group[action_column]),
                        period=period,
                        arm=policy_name,
                        policy=policy_name,
                        scope=f"{state}|{phase}",
                    )
                    records.append(
                        {
                            **metrics,
                            "market_shock_state_v1": state,
                            "tail_phase_v1": phase,
                            "phase_population": (
                                "persistent_checkpoint_rows_descriptive_not_independent"
                                if phase == "PERSISTENT"
                                else "fresh_tail_entries"
                            ),
                            "support_status": support_status,
                            "checkpoint_distribution_json": _distribution(
                                group["checkpoint"]
                            ),
                        }
                    )
    return pd.DataFrame(records)


def _shock_sign_stratification(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for period in ("assessment", "stress"):
        for state in ONSET_STATES:
            state_rows = frame.loc[
                frame["partition"].eq(period)
                & frame["market_shock_state_v1"].eq(state)
            ]
            for arm, response_class, aligned_column in (
                (
                    "continuation",
                    "AMPLIFYING",
                    "continuation_aligned_return_v1",
                ),
                ("resistance", "RESISTING", "resistance_aligned_return_v1"),
            ):
                group = state_rows.loc[
                    state_rows["shock_response_class_v1"].eq(response_class)
                    & state_rows["primary_outcome_complete_v1"].astype(bool)
                ]
                predicts_up = (
                    arm == "continuation" and state.startswith("POSITIVE")
                ) or (arm == "resistance" and state.startswith("NEGATIVE"))
                predicted_material_state = (
                    "MATERIAL_UP" if predicts_up else "MATERIAL_DOWN"
                )
                records.append(
                    {
                        "period": period,
                        "shock_state": state,
                        "arm": arm,
                        "episode_count": int(len(group)),
                        "session_count": int(group["session"].nunique()),
                        "unique_shock_event_count": int(
                            group["market_shock_event_id_v1"].nunique()
                        ),
                        "stock_count": int(group["stock"].nunique()),
                        "mean_aligned_return": (
                            float(group[aligned_column].mean())
                            if len(group)
                            else math.nan
                        ),
                        "material_direction_accuracy": (
                            float(
                                (
                                    group["primary_outcome_state_v1"].eq(
                                        predicted_material_state
                                    )
                                )
                                .loc[
                                    group["primary_outcome_state_v1"].isin(
                                        ["MATERIAL_UP", "MATERIAL_DOWN"]
                                    )
                                ]
                                .mean()
                            )
                            if group["primary_outcome_state_v1"]
                            .isin(["MATERIAL_UP", "MATERIAL_DOWN"])
                            .any()
                            else math.nan
                        ),
                    }
                )
    return pd.DataFrame(records)


def _missingness_table(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for period in ("development", "assessment", "stress"):
        working = frame.loc[frame["partition"].eq(period)]
        reason_columns = (
            ("market_window", "market_window_missing_reasons_v1"),
            ("market_shock", "market_shock_missing_reasons_v1"),
            ("stock_response", "shock_response_missing_reasons_v1"),
        )
        for category, column in reason_columns:
            counts: dict[str, int] = {}
            for raw in working[column]:
                reasons = (
                    tuple(raw)
                    if isinstance(raw, (list, tuple, np.ndarray))
                    else (() if pd.isna(raw) else (str(raw),))
                )
                for reason in reasons:
                    counts[str(reason)] = counts.get(str(reason), 0) + 1
            if not counts:
                records.append(
                    {
                        "period": period,
                        "category": category,
                        "reason": "none",
                        "episode_count": 0,
                    }
                )
            for reason, count in sorted(counts.items()):
                records.append(
                    {
                        "period": period,
                        "category": category,
                        "reason": reason,
                        "episode_count": count,
                    }
                )
    return pd.DataFrame(records)


def _lookup_bootstrap_lower(
    summary: pd.DataFrame,
    *,
    period: str,
    cluster_type: str,
    statistic: str,
) -> float:
    selected = summary.loc[
        summary["period"].eq(period)
        & summary["cluster_type"].eq(cluster_type)
        & summary["statistic"].eq(statistic),
        "lower_95",
    ]
    return float(selected.iloc[0]) if len(selected) else math.nan


def _lookup_policy(
    baselines: pd.DataFrame,
    *,
    period: str,
    scope: str,
    policy: str,
    metric: str,
) -> float:
    selected = baselines.loc[
        baselines["period"].eq(period)
        & baselines["evaluation_scope"].eq(scope)
        & baselines["policy"].eq(policy),
        metric,
    ]
    if not len(selected) or pd.isna(selected.iloc[0]):
        return math.nan
    return float(selected.iloc[0])


def _bootstrap_interval(
    summary: pd.DataFrame,
    *,
    period: str,
    cluster_type: str,
    statistic: str,
) -> tuple[float, float]:
    selected = summary.loc[
        summary["period"].eq(period)
        & summary["cluster_type"].eq(cluster_type)
        & summary["statistic"].eq(statistic)
    ]
    if len(selected) != 1:
        return math.nan, math.nan
    return float(selected.iloc[0]["lower_95"]), float(selected.iloc[0]["upper_95"])


def _enrich_primary_results(
    *,
    continuation: pd.DataFrame,
    resistance: pd.DataFrame,
    ranking: pd.DataFrame,
    baselines: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
) -> None:
    for period in ("assessment", "stress"):
        for table, arm, statistic in (
            (
                continuation,
                "continuation",
                "continuation_mean_aligned_return",
            ),
            (
                resistance,
                "resistance",
                "resistance_mean_aligned_return",
            ),
        ):
            row_mask = table["period"].eq(period)
            for prefix, cluster in (
                ("session_cluster", "session"),
                ("shock_event_cluster", "market_shock_event_id_v1"),
            ):
                lower, upper = _bootstrap_interval(
                    bootstrap_summary,
                    period=period,
                    cluster_type=cluster,
                    statistic=statistic,
                )
                table.loc[row_mask, f"{prefix}_lower_95"] = lower
                table.loc[row_mask, f"{prefix}_upper_95"] = upper
            scope = (
                "continuation_amplifying_acted"
                if arm == "continuation"
                else "resistance_resisting_acted"
            )
            same_follow = _lookup_policy(
                baselines,
                period=period,
                scope=scope,
                policy="follow_market_shock",
                metric="mean_aligned_return",
            )
            table.loc[row_mask, "follow_market_same_acted_mean_aligned_return"] = (
                same_follow
            )
            if arm == "continuation":
                all_follow_mean = _lookup_policy(
                    baselines,
                    period=period,
                    scope="all_signed_shock_onsets",
                    policy="follow_market_shock",
                    metric="mean_aligned_return",
                )
                all_follow_accuracy = _lookup_policy(
                    baselines,
                    period=period,
                    scope="all_signed_shock_onsets",
                    policy="follow_market_shock",
                    metric="accuracy_counting_no_move_as_failure",
                )
                selected_mean = float(
                    table.loc[row_mask, "mean_aligned_return"].iloc[0]
                )
                selected_accuracy = float(
                    table.loc[
                        row_mask,
                        "accuracy_counting_no_move_as_failure",
                    ].iloc[0]
                )
                table.loc[row_mask, "all_shock_follow_mean_aligned_return"] = (
                    all_follow_mean
                )
                table.loc[
                    row_mask,
                    "amplifying_selection_mean_return_delta_vs_all_shock_follow",
                ] = selected_mean - all_follow_mean
                table.loc[
                    row_mask,
                    "all_shock_follow_accuracy_counting_no_move_as_failure",
                ] = all_follow_accuracy
                table.loc[
                    row_mask,
                    "amplifying_selection_accuracy_delta_vs_all_shock_follow",
                ] = selected_accuracy - all_follow_accuracy
            else:
                selected_mean = float(
                    table.loc[row_mask, "mean_aligned_return"].iloc[0]
                )
                table.loc[
                    row_mask,
                    "resistance_minus_follow_market_same_acted_mean_return",
                ] = selected_mean - same_follow
                for policy, label in (
                    ("recent_stock_momentum_5m", "recent_momentum"),
                    ("frozen_A1", "frozen_a1"),
                ):
                    value = _lookup_policy(
                        baselines,
                        period=period,
                        scope=scope,
                        policy=policy,
                        metric="mean_aligned_return",
                    )
                    table.loc[
                        row_mask,
                        f"{label}_same_acted_mean_aligned_return",
                    ] = value
                    table.loc[
                        row_mask,
                        f"resistance_minus_{label}_mean_return",
                    ] = selected_mean - value

        rank_mask = ranking["period"].eq(period)
        for prefix, cluster in (
            ("session_cluster", "session"),
            ("shock_event_cluster", "market_shock_event_id_v1"),
        ):
            lower, upper = _bootstrap_interval(
                bootstrap_summary,
                period=period,
                cluster_type=cluster,
                statistic="continuous_ranking_auc",
            )
            ranking.loc[rank_mask, f"{prefix}_auc_lower_95"] = lower
            ranking.loc[rank_mask, f"{prefix}_auc_upper_95"] = upper


def _leave_one_dependency_pass(
    leave_tables: Sequence[pd.DataFrame],
    *,
    arm: str,
) -> bool:
    for period in ("assessment", "stress"):
        relevant = pd.concat(
            [
                table.loc[
                    table["period"].eq(period) & table["arm"].eq(arm)
                ]
                for table in leave_tables
            ],
            ignore_index=True,
        )
        values = pd.to_numeric(relevant["mean_aligned_return"], errors="coerce")
        if not (len(values) and values.notna().all() and values.gt(0.0).all()):
            return False
    return True


def _decision_contract(
    *,
    continuation: pd.DataFrame,
    resistance: pd.DataFrame,
    ranking: pd.DataFrame,
    baselines: pd.DataFrame,
    sign_results: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
    holm: Mapping[str, float],
    leave_tables: Sequence[pd.DataFrame],
    normal_comparison: pd.DataFrame,
) -> dict[str, Any]:
    by_arm = {
        "continuation": continuation.set_index("period"),
        "resistance": resistance.set_index("period"),
    }
    evidence: dict[str, dict[str, Any]] = {}
    for arm in ("continuation", "resistance"):
        table = by_arm[arm]
        statistic = f"{arm}_mean_aligned_return"
        scope = (
            "continuation_amplifying_acted"
            if arm == "continuation"
            else "resistance_resisting_acted"
        )
        support_assessment = bool(table.loc["assessment", "support_pass"])
        support_stress = bool(table.loc["stress", "support_pass"])
        means_positive = bool(
            (table.loc[["assessment", "stress"], "mean_aligned_return"] > 0.0).all()
        )
        session_lower = _lookup_bootstrap_lower(
            bootstrap_summary,
            period="assessment",
            cluster_type="session",
            statistic=statistic,
        )
        event_lower = _lookup_bootstrap_lower(
            bootstrap_summary,
            period="assessment",
            cluster_type="market_shock_event_id_v1",
            statistic=statistic,
        )
        accuracy_both = bool(
            (
                table.loc[
                    ["assessment", "stress"],
                    "material_direction_accuracy",
                ]
                > 0.50
            ).all()
        )
        following_exceeds_opposing = bool(
            (
                table.loc[
                    ["assessment", "stress"],
                    "material_following_minus_opposing_rate",
                ]
                > 0.0
            ).all()
        )
        sign_subset = sign_results.loc[sign_results["arm"].eq(arm)]
        same_sign = bool(
            len(sign_subset) == 4
            and sign_subset["mean_aligned_return"].notna().all()
            and sign_subset["mean_aligned_return"].gt(0.0).all()
        )
        not_dependent = _leave_one_dependency_pass(leave_tables, arm=arm)
        if arm == "continuation":
            policy_no_move = {
                period: _lookup_policy(
                    baselines,
                    period=period,
                    scope=scope,
                    policy="continuation_v1",
                    metric="accuracy_counting_no_move_as_failure",
                )
                for period in ("assessment", "stress")
            }
            all_follow = {
                period: _lookup_policy(
                    baselines,
                    period=period,
                    scope="all_signed_shock_onsets",
                    policy="follow_market_shock",
                    metric="accuracy_counting_no_move_as_failure",
                )
                for period in ("assessment", "stress")
            }
            baseline_pass = all(
                policy_no_move[period] > all_follow[period]
                for period in ("assessment", "stress")
            )
        else:
            opposite = {
                period: _lookup_policy(
                    baselines,
                    period=period,
                    scope=scope,
                    policy="resistance_v1",
                    metric="mean_aligned_return",
                )
                for period in ("assessment", "stress")
            }
            follow = {
                period: _lookup_policy(
                    baselines,
                    period=period,
                    scope=scope,
                    policy="follow_market_shock",
                    metric="mean_aligned_return",
                )
                for period in ("assessment", "stress")
            }
            baseline_pass = all(
                opposite[period] > follow[period]
                for period in ("assessment", "stress")
            )
        tests = {
            "assessment_support_pass": support_assessment,
            "stress_support_pass": support_stress,
            "mean_aligned_return_positive_both_periods": means_positive,
            "assessment_session_cluster_lower_95_above_zero": session_lower > 0.0,
            "assessment_shock_event_cluster_lower_95_above_zero": event_lower > 0.0,
            "material_predicted_rate_exceeds_opposing_both_periods": (
                following_exceeds_opposing
            ),
            "material_accuracy_above_half_both_periods": accuracy_both,
            "required_baseline_comparison_pass": baseline_pass,
            "same_broad_sign_positive_and_negative_shocks": same_sign,
            "holm_adjusted_assessment_null_p_below_0_05": (
                math.isfinite(holm[arm]) and holm[arm] < 0.05
            ),
            "not_dependent_on_single_cluster": not_dependent,
        }
        evidence[arm] = {
            "tests": tests,
            "full_contract_pass": bool(all(tests.values())),
            "assessment_session_cluster_lower_95": session_lower,
            "assessment_shock_event_cluster_lower_95": event_lower,
            "holm_adjusted_assessment_null_p": holm[arm],
        }

    rank = ranking.set_index("period")
    rank_reproducible = bool(
        len(rank) == 2
        and rank.loc["assessment", "roc_auc_followed_shock_among_material_movers"]
        > 0.50
        and rank.loc["stress", "roc_auc_followed_shock_among_material_movers"] > 0.50
        and rank.loc["assessment", "session_cluster_auc_lower_95"] > 0.50
        and rank.loc["assessment", "shock_event_cluster_auc_lower_95"] > 0.50
    )
    continuation_support_statuses = set(continuation["support_status"].astype(str))
    if evidence["continuation"]["full_contract_pass"]:
        continuation_decision = "shock_continuation_supported_retrospectively"
    elif rank_reproducible:
        continuation_decision = "shock_continuation_ranking_only"
    elif "blocked_insufficient_independent_shock_support" in continuation_support_statuses:
        continuation_decision = "blocked_insufficient_independent_shock_support"
    elif not continuation["support_pass"].astype(bool).all():
        continuation_decision = "blocked_insufficient_support"
    else:
        continuation_decision = "no_incremental_signed_shock_directional_signal"

    resistance_support_statuses = set(resistance["support_status"].astype(str))
    if evidence["resistance"]["full_contract_pass"]:
        resistance_decision = "shock_resistance_supported_retrospectively"
    elif "blocked_insufficient_independent_shock_support" in resistance_support_statuses:
        resistance_decision = "blocked_insufficient_independent_shock_support"
    elif not resistance["support_pass"].astype(bool).all():
        resistance_decision = "blocked_insufficient_support"
    else:
        resistance_decision = "shock_resistance_descriptive_only"

    if evidence["continuation"]["full_contract_pass"] or evidence["resistance"][
        "full_contract_pass"
    ]:
        overall = "signed_market_transition_direction_supported_retrospectively"
    else:
        standardised = normal_comparison.loc[
            normal_comparison["aggregation"].eq(
                "equal_weight_common_checkpoint_standardised"
            )
        ]
        movement_only = True
        for period in ("assessment", "stress"):
            shock = standardised.loc[
                standardised["period"].eq(period)
                & standardised["regime"].eq("SIGNED_SHOCK_ONSET")
            ]
            normal = standardised.loc[
                standardised["period"].eq(period)
                & standardised["regime"].eq("NORMAL_OTHER")
            ]
            movement_only = (
                movement_only
                and len(shock) == 1
                and len(normal) == 1
                and float(shock.iloc[0]["mean_absolute_15m_movement"])
                > float(normal.iloc[0]["mean_absolute_15m_movement"])
            )
        if movement_only:
            overall = "signed_shock_movement_only"
        elif (
            continuation_decision.startswith("blocked_")
            and resistance_decision.startswith("blocked_")
        ):
            overall = (
                "blocked_insufficient_independent_shock_support"
                if "independent" in continuation_decision + resistance_decision
                else "blocked_insufficient_support"
            )
        else:
            overall = "no_incremental_signed_shock_directional_signal"
    return {
        "continuation_decision": continuation_decision,
        "resistance_decision": resistance_decision,
        "overall_decision": overall,
        "continuation_contract": evidence["continuation"],
        "resistance_contract": evidence["resistance"],
        "continuous_ranking_reproducible_above_half": rank_reproducible,
        "retrospective_opened_periods": True,
        "untouched_confirmation": False,
        "option_edge_claimed": False,
        "tradeability_claimed": False,
    }


def _calibration_table(
    thresholds: Mapping[int, CheckpointShockThresholdsV1],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            threshold.model_dump(mode="python")
            for _, threshold in sorted(thresholds.items())
        ]
    )


def _threshold_manifest(
    thresholds: Mapping[int, CheckpointShockThresholdsV1],
    *,
    configuration_hash: str,
) -> dict[str, Any]:
    return {
        "schema_version": "m1c-signed-market-shock-thresholds-v1",
        "market_proxy_v1": MARKET_SHOCK_PROXY_V1,
        "calibration_period": {
            "start": DEVELOPMENT_START,
            "end": DEVELOPMENT_END,
            "predictors_only": True,
            "future_stock_outcomes_accessed_for_thresholds": False,
            "option_outcomes_accessed_for_thresholds": False,
        },
        "quantiles": {
            "signed_return_lower": 0.10,
            "signed_return_upper": 0.90,
            "range": 0.75,
            "method": "numpy_linear",
        },
        "minimum_predictor_support_v1": MINIMUM_PREDICTOR_SUPPORT_V1,
        "pooling_fallback_used": False,
        "configuration_hash": configuration_hash,
        "checkpoints": [
            threshold.model_dump(mode="json")
            for _, threshold in sorted(thresholds.items())
        ],
    }


def _market_proxy_audit(
    *,
    alignment: Mapping[str, Any],
    vti: pd.DataFrame,
    market_states: pd.DataFrame,
    bounded_hash: str,
) -> dict[str, Any]:
    missing_by_checkpoint = (
        market_states.groupby("checkpoint", sort=True)["market_window_complete_v1"]
        .agg(complete_rows="sum", scheduled_rows="size")
        .reset_index()
    )
    missing_by_checkpoint["incomplete_rows"] = (
        missing_by_checkpoint["scheduled_rows"]
        - missing_by_checkpoint["complete_rows"]
    )
    return {
        "canonical_proxy_available": True,
        "proxy_identifier": MARKET_SHOCK_PROXY_V1,
        "source": "EODHD canonical processed stock bars",
        "source_path": str(VTI_PATH),
        "frequency": "five_minutes",
        "timestamp_semantics": (
            "UTC bar-start timestamp; a bar is causal only at bar_start_plus_5_minutes"
        ),
        "trading_calendar": "NYSE regular session 09:30-16:00 America/New_York",
        "adjustment_convention": (
            "raw/unadjusted OHLC; source has no adjusted-close field"
        ),
        "imputation": False,
        "partial_bars_allowed": False,
        "session_crossing_allowed": False,
        "bounded_opened_row_count": int(len(vti)),
        "bounded_min_timestamp": pd.to_datetime(vti["timestamp"], utc=True).min(),
        "bounded_max_timestamp": pd.to_datetime(vti["timestamp"], utc=True).max(),
        "bounded_arrow_sha256": bounded_hash,
        "bounded_rows_with_missing_ohlc_count": int(
            vti[["open", "high", "low", "close"]].isna().any(axis=1).sum()
        ),
        "protected_filter": "timestamp >= 2024-01-01T00:00:00Z and < 2026-01-01T00:00:00Z",
        "existing_causal_alignment": dict(alignment),
        "bars_at_frozen_checkpoints": missing_by_checkpoint.to_dict(orient="records"),
        "checkpoint_6_note": (
            "VTI bars are present, but W1 is deterministically incomplete because "
            "its reference would cross the session boundary"
        ),
        "alternative_proxies_tested": False,
        "new_external_dataset_acquired": False,
    }


def _selected_episode_columns(frame: pd.DataFrame) -> list[str]:
    requested = [
        "row_id",
        "episode_id",
        "existing_fresh_episode_identifier",
        "stock",
        "session",
        "partition",
        "checkpoint",
        "checkpoint_group_v1",
        "time_of_day_v1",
        "feature_available_timestamp_utc",
        "signal_timestamp",
        "market_signal_timestamp_v1",
        "prospective_entry_timestamp",
        "M1C_probability",
        "m1c_high_tail_threshold_v1",
        "m1c_high_tail_v1",
        "m1c_model_version_v1",
        "m1c_model_hash_v1",
        "m1c_feature_hash_v1",
        "m1c_tail_phase_v1",
        "tail_entry_number_v1",
        "tail_run_length_checkpoints_v1",
        "tail_run_age_minutes_v1",
        "movement_consumed_v1",
        "A1_probability_up_v1",
        "A1_action_v1",
        "A1_model_hash_v1",
        "market_proxy_v1",
        "w0_bar_ordinals_v1",
        "w1_bar_ordinals_v1",
        "market_return_w0_v1",
        "market_range_w0_v1",
        "market_return_w1_v1",
        "market_range_w1_v1",
        "maximum_market_timestamp_v1",
        "market_window_complete_v1",
        "market_window_missing_reasons_v1",
        "market_shock_state_v1",
        "market_shock_event_id_v1",
        "shock_sign_v1",
        "market_shock_complete_v1",
        "market_shock_missing_reasons_v1",
        "threshold_market_return_w0_q10_v1",
        "threshold_market_return_w0_q90_v1",
        "threshold_market_range_w0_q75_v1",
        "threshold_market_return_w1_q10_v1",
        "threshold_market_return_w1_q90_v1",
        "threshold_market_range_w1_q75_v1",
        "threshold_calibration_complete_v1",
        "threshold_calibration_missing_reason_v1",
        "threshold_15m",
        "stock_return_w0_v1",
        "stock_absolute_alignment_v1",
        "shock_relative_response_v1",
        "shock_response_class_v1",
        "resisting_subtype_v1",
        "maximum_stock_timestamp_v1",
        "shock_response_complete_v1",
        "shock_response_missing_reasons_v1",
        "recent_stock_return_5m_v1",
        "future_signed_return_15m",
        "future_absolute_movement_15m_v1",
        "primary_outcome_state_v1",
        "primary_outcome_complete_v1",
        "future_iv_residual_15m_v1",
        "future_exceed_iv_15m_v1",
        "post_share_of_local_range_v1",
        "continuation_aligned_return_v1",
        "resistance_aligned_return_v1",
        "followed_shock_v1",
        "continuation_action_v1",
        "resistance_action_v1",
        "response_quintile_v1",
    ]
    return [column for column in requested if column in frame.columns]


def _format_number(value: Any, digits: int = 6) -> str:
    if value is None or pd.isna(value):
        return "NA"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    return f"{float(value):.{digits}f}"


def _markdown_table(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
) -> str:
    selected = frame.loc[:, list(columns) if columns is not None else frame.columns]
    headers = [str(column) for column in selected.columns]

    def render(value: Any) -> str:
        if value is None or value is pd.NA:
            return "NA"
        try:
            if bool(pd.isna(value)):
                return "NA"
        except (TypeError, ValueError):
            pass
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6g}"
        text = str(value).replace("\n", " ").replace("|", "\\|")
        return text

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(render(value) for value in row) + " |"
        for row in selected.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def _report_markdown(
    *,
    proxy_audit: Mapping[str, Any],
    calibration: pd.DataFrame,
    accounting: pd.DataFrame,
    assessment_outcomes: pd.DataFrame,
    stress_outcomes: pd.DataFrame,
    continuation: pd.DataFrame,
    resistance: pd.DataFrame,
    ranking: pd.DataFrame,
    baselines: pd.DataFrame,
    normal: pd.DataFrame,
    sign_results: pd.DataFrame,
    decisions: Mapping[str, Any],
    null_summary: Mapping[str, Any],
    tail: pd.DataFrame,
    missingness: pd.DataFrame,
) -> str:
    onset_accounting = accounting.set_index("period")
    continuation_by_period = continuation.set_index("period")
    resistance_by_period = resistance.set_index("period")
    ranking_by_period = ranking.set_index("period")
    ranking_auc_column = "roc_auc_followed_shock_among_material_movers"
    assessment_auc = ranking_by_period.at["assessment", ranking_auc_column]
    stress_auc = ranking_by_period.at["stress", ranking_auc_column]
    alignment = cast(
        Mapping[str, Any],
        proxy_audit["existing_causal_alignment"],
    )
    standardised = normal.loc[
        normal["aggregation"].eq("equal_weight_common_checkpoint_standardised")
    ]
    lines = [
        "# M1C Signed Market Shock Transition V1",
        "",
        "Research-only retrospective report. The 2025 assessment and stress periods "
        "were opened by earlier research; this is out-of-development evidence, not "
        "untouched confirmation.",
        "",
        "## Decisions",
        "",
        f"- Continuation: `{decisions['continuation_decision']}`",
        f"- Resistance: `{decisions['resistance_decision']}`",
        f"- Overall: `{decisions['overall_decision']}`",
        "- Option profitability: **Not tested**",
        "- Tradeability: not claimed",
        "",
        "## Structural regime evidence",
        "",
        "A canonical causal market proxy was available: **VTI**, from the existing "
        "EODHD five-minute market-direction baseline. Bars are raw/unadjusted OHLC, "
        "timestamped at bar start in UTC, final only five minutes later, and aligned "
        "to the NYSE regular-session calendar. No alternative proxy was tested.",
        "",
        f"The bounded proxy audit compared {alignment['rows_compared']:,} "
        "archived causal bar observations; maximum return and range differences were "
        f"{alignment['maximum_absolute_bar_return_difference']:.3g} "
        "and "
        f"{alignment['maximum_absolute_bar_range_difference']:.3g}.",
        "",
        "Checkpoint 6 is `UNKNOWN_INCOMPLETE`: W1 would require a previous-session "
        "reference. It was not pooled, imputed, or given a fallback.",
        "",
        _markdown_table(
            accounting,
            columns=[
                "period",
                "sessions_with_complete_market_data",
                "scheduled_sessions",
                "unique_negative_shock_onsets",
                "unique_positive_shock_onsets",
                "ongoing_shocks",
                "elevated_range_nondirectional_states",
                "unique_market_shock_event_ids",
                "mean_stocks_per_shock_event",
                "incomplete_fresh_episodes",
            ],
        ),
        "",
        "### Exact fixed market windows",
        "",
        "- W0 uses market bars `checkpoint-3` through `checkpoint-1`, with return "
        "`log(close[checkpoint-1] / close[checkpoint-4])`.",
        "- W1 uses market bars `checkpoint-6` through `checkpoint-4`, with return "
        "`log(close[checkpoint-4] / close[checkpoint-7])`.",
        "- Ranges are `log(max(high) / min(low))` within the respective three bars.",
        "- Both windows end at or before the M1C signal; neither contains the "
        "next-bar-open entry bar.",
        "",
        "### Frozen 2024 predictor-only thresholds",
        "",
        _markdown_table(calibration),
        "",
        "The definitions use inclusive 10th/90th signed-return tails and the "
        "inclusive 75th-percentile range boundary. Current shocks absent the same "
        "prior shock are onsets; repeated same-signed shocks are ongoing. Elevated "
        "range without a signed tail is nondirectional; all other complete rows are "
        "normal.",
        "",
        "## Absolute-movement evidence",
        "",
    ]
    for period in ("assessment", "stress"):
        selected = standardised.loc[standardised["period"].eq(period)].set_index(
            "regime"
        )
        if {"SIGNED_SHOCK_ONSET", "NORMAL_OTHER"}.issubset(selected.index):
            shock = selected.loc["SIGNED_SHOCK_ONSET"]
            ordinary = selected.loc["NORMAL_OTHER"]
            lines.extend(
                [
                    f"- {period.title()}: checkpoint-standardised shock no-move "
                    f"{_format_number(shock['no_move_rate'])} versus normal "
                    f"{_format_number(ordinary['no_move_rate'])}; IV-excess "
                    f"{_format_number(shock['iv_excess_rate'])} versus "
                    f"{_format_number(ordinary['iv_excess_rate'])}; mean absolute "
                    f"movement {_format_number(shock['mean_absolute_15m_movement'])} "
                    f"versus {_format_number(ordinary['mean_absolute_15m_movement'])}.",
                ]
            )
    lines.extend(
        [
            "",
            "No-move outcomes remain in every action-policy denominator where "
            "specified.",
            "",
            "## Directional evidence",
            "",
            "The stock response is fixed as "
            "`shock_sign × (stock_return_w0 - market_return_w0) / threshold_15m`. "
            "Positive is AMPLIFYING; negative is RESISTING; exact zero is "
            "NEUTRAL_EXACT. The resistance and continuation policies remain separate.",
            "The assessment/stress outcome artifacts include both descriptive "
            "RESISTING subtypes and every response class within each shock sign; "
            "`checkpoint_stratified_mechanism_results_v1.csv` reports every frozen "
            "checkpoint without selection.",
            "",
            "### Continuation arm",
            "",
            _markdown_table(
                continuation,
                columns=[
                    "period",
                    "acted_episode_count",
                    "unique_shock_event_count",
                    "session_count",
                    "stock_count",
                    "call_count",
                    "put_count",
                    "no_material_move_count",
                    "material_direction_accuracy",
                    "accuracy_counting_no_move_as_failure",
                    "mean_aligned_return",
                    "session_cluster_lower_95",
                    "shock_event_cluster_lower_95",
                    "one_percent_winsorised_mean_aligned_return",
                    "all_shock_follow_mean_aligned_return",
                    "amplifying_selection_mean_return_delta_vs_all_shock_follow",
                    "positive_session_rate",
                    "positive_month_rate",
                    "support_status",
                    "decision",
                ],
            ),
            "",
            "### Resistance arm",
            "",
            _markdown_table(
                resistance,
                columns=[
                    "period",
                    "acted_episode_count",
                    "unique_shock_event_count",
                    "session_count",
                    "stock_count",
                    "call_count",
                    "put_count",
                    "no_material_move_count",
                    "material_direction_accuracy",
                    "accuracy_counting_no_move_as_failure",
                    "mean_aligned_return",
                    "session_cluster_lower_95",
                    "shock_event_cluster_lower_95",
                    "one_percent_winsorised_mean_aligned_return",
                    "follow_market_same_acted_mean_aligned_return",
                    "resistance_minus_follow_market_same_acted_mean_return",
                    "positive_session_rate",
                    "positive_month_rate",
                    "support_status",
                    "decision",
                ],
            ),
            "",
            "### Continuous ranking",
            "",
            _markdown_table(ranking),
            "",
            "### Sign consistency",
            "",
            _markdown_table(sign_results),
            "",
            "### Fixed baselines",
            "",
            _markdown_table(
                baselines.loc[
                    baselines["evaluation_scope"].isin(
                        [
                            "all_signed_shock_onsets",
                            "continuation_amplifying_acted",
                            "resistance_resisting_acted",
                        ]
                    )
                ],
                columns=[
                    "period",
                    "evaluation_scope",
                    "policy",
                    "acted_episode_count",
                    "unique_shock_event_count",
                    "material_direction_accuracy",
                    "accuracy_counting_no_move_as_failure",
                    "mean_aligned_return",
                    "status",
                ],
            ),
            "",
            "Frozen D2 is blocked because its clean reproducible lineage is not "
            "available; it was not approximately reconstructed.",
            "",
            "### Null and placebo",
            "",
            "The primary null reassigns each outcome within stock, checkpoint, and "
            "period to a different session. It used 1,000 fixed-seed replications. "
            "The temporal placebo uses the next eligible fresh high-M1C outcome for the "
            "same stock/checkpoint/period without crossing chronology.",
            "",
            "```json",
            json.dumps(_json_safe(dict(null_summary)), indent=2, sort_keys=True),
            "```",
            "",
            "Both session-cluster and shared-shock-event-cluster uncertainty govern "
            "the decisions; the more conservative conclusion is used.",
            "",
            "## Normal-regime comparison",
            "",
            _markdown_table(
                normal.loc[
                    normal["aggregation"].isin(
                        ["raw", "equal_weight_common_checkpoint_standardised"]
                    )
                ],
                columns=[
                    "period",
                    "regime",
                    "aggregation",
                    "episode_count",
                    "session_count",
                    "checkpoint_count",
                    "conditional_direction_accuracy_among_material_movers",
                    "accuracy_counting_no_move_as_failure",
                    "mean_market_aligned_15m_return",
                    "no_move_rate",
                    "iv_excess_rate",
                    "mean_absolute_15m_movement",
                ],
            ),
            "",
            "This is an exact/equal-weight checkpoint adjustment, not a fitted model "
            "and not favourable-checkpoint selection.",
            "",
            "## Tail Phase evidence",
            "",
            "Tail Phase V1 is attached unchanged and is descriptive only. FIRST_ENTRY "
            "and RE_ENTRY retain fresh-episode interpretation. PERSISTENT rows are "
            "labelled as dependent checkpoint observations and never counted as "
            "independent primary episode support. No phase gate or interaction was "
            "fitted.",
            "",
            _markdown_table(
                tail,
                columns=[
                    "period",
                    "market_shock_state_v1",
                    "tail_phase_v1",
                    "policy",
                    "eligible_episode_count",
                    "acted_episode_count",
                    "unique_shock_event_count",
                    "material_direction_accuracy",
                    "mean_aligned_return",
                    "support_status",
                    "phase_population",
                    "checkpoint_distribution_json",
                ],
            ),
            "",
            "## Data completeness and operational blockers",
            "",
            _markdown_table(missingness),
            "",
            "Missing windows, calibration, timestamps, stock bars, or IV scales fail "
            "closed as `UNKNOWN_INCOMPLETE`. Operational blockers are not interpreted "
            "as negative scientific evidence.",
            "",
            "## Execution realism",
            "",
            "Five-minute historical bars cannot observe bid withdrawal, ask "
            "withdrawal, replenishment, trade impact, spread changes, queue behaviour, "
            "or executable option outcomes. Prospective bid/ask and trade-impact "
            "recording is required to learn those quantities.",
            "",
            "No option P&L, hypothetical option return, midpoint fill, broker access, "
            "order routing, or order placement occurred.",
            "",
            "## Direct answers",
            "",
            "1. **Canonical proxy?** Yes—VTI, already used by the causal baseline.",
            f"2. **Onset frequency?** Assessment: "
            f"{int(onset_accounting.loc['assessment', 'unique_negative_shock_onsets'])} "
            "negative and "
            f"{int(onset_accounting.loc['assessment', 'unique_positive_shock_onsets'])} "
            f"positive; stress: "
            f"{int(onset_accounting.loc['stress', 'unique_negative_shock_onsets'])} "
            "negative and "
            f"{int(onset_accounting.loc['stress', 'unique_positive_shock_onsets'])} positive.",
            "3. **Spread?** Session, checkpoint, stock, and shock-event support and "
            "concentration are reported explicitly; decision caps were enforced.",
            "4. **Movement difference?** The checkpoint-standardised no-move, "
            "IV-excess, and absolute-movement comparisons above answer this without "
            "directional selection.",
            f"5. **Amplifying continuation?** `{decisions['continuation_decision']}`; "
            "assessment/stress mean aligned returns were "
            f"{_format_number(continuation_by_period.loc['assessment', 'mean_aligned_return'])}/"
            f"{_format_number(continuation_by_period.loc['stress', 'mean_aligned_return'])}.",
            f"6. **Resisting opposition?** `{decisions['resistance_decision']}`; "
            "assessment/stress mean aligned returns were "
            f"{_format_number(resistance_by_period.loc['assessment', 'mean_aligned_return'])}/"
            f"{_format_number(resistance_by_period.loc['stress', 'mean_aligned_return'])}.",
            "7. **Continuous rank?** Assessment/stress AUCs were "
            f"{_format_number(assessment_auc)}/{_format_number(stress_auc)}.",
            "8. **Assessment and stress unchanged?** Frozen 2024 thresholds and "
            "response definitions were applied unchanged to both.",
            "9. **Baselines?** The fixed same-timestamp comparisons above include "
            "market direction, opposition, recent 5m and trailing 15m momentum, "
            "frozen A1, always CALL/PUT, and blocked D2.",
            "10. **Clustering?** Both session and shared-shock-event bootstraps were "
            "required by the decision contract.",
            "11. **Both shock signs?** Separate positive/negative results are shown "
            "above and are contract inputs.",
            "12. **Dominance?** Leave-one-out and concentration artifacts cover "
            "stock, month, checkpoint, session, and shock event.",
            "13. **Tail Phase?** Descriptive only; it did not alter any action.",
            "14. **Easier than normal?** See the checkpoint-standardised normal-regime "
            "table; no model or checkpoint selection was used.",
            f"15. **Classification?** `{decisions['overall_decision']}`.",
            "16. **Still unknowable?** Executable option prices, liquidity withdrawal/"
            "replenishment, impact, spreads, queueing, fills, and prospective "
            "behavioural stability.",
            "",
            "## Frozen-system confirmations",
            "",
            "- M1C probabilities, threshold, horizon, high-tail membership, and fresh "
            "episode identifiers were unchanged.",
            "- Tail Phase V1 and frozen A1 were unchanged.",
            "- Archived signed pressure, archived tension, contaminated descendants, "
            "future peer slates, and cross-sectional normalisation were not used.",
            "- No protected 2026 historical outcome was opened, calculated, displayed, "
            "or inspected.",
            "- No broker was accessed; no order-routing path was enabled; no order was "
            "placed.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_artifacts(
    *,
    payloads: Mapping[str, tuple[str, Any]],
) -> dict[str, str]:
    paths: dict[str, str] = {}
    for filename, (kind, payload) in payloads.items():
        path = PRIMARY / filename
        if kind == "json":
            _write_json(path, cast(Mapping[str, Any], payload))
        elif kind == "csv":
            _write_csv(path, cast(pd.DataFrame, payload))
        elif kind == "parquet":
            _write_parquet(path, cast(pd.DataFrame, payload))
        else:
            raise ValueError(f"unknown artifact kind: {kind}")
        paths[filename] = _sha256_file(path)
    return paths


def _validate_before_write(
    *,
    episodes: pd.DataFrame,
    market_states: pd.DataFrame,
    events: pd.DataFrame,
    thresholds: Mapping[int, CheckpointShockThresholdsV1],
    session_bootstrap: pd.DataFrame,
    event_bootstrap: pd.DataFrame,
    null_draws: pd.DataFrame,
) -> None:
    assert_unprotected_sessions_v1(episodes["session"])
    assert_unprotected_sessions_v1(market_states["session"])
    if episodes.duplicated(IDENTITY).any():
        raise ExperimentBlocked("episode output identities duplicated")
    if market_states.duplicated(["session", "checkpoint"]).any():
        raise ExperimentBlocked("market-state surface identities duplicated")
    if events["market_shock_event_id_v1"].duplicated().any():
        raise ExperimentBlocked("shock-event output identities duplicated")
    event_components = events[
        ["session", "checkpoint", "market_proxy_v1", "shock_sign_v1"]
    ]
    if event_components.duplicated().any():
        raise ExperimentBlocked("two event identifiers represent one shock")
    onset = episodes.loc[episodes["market_shock_state_v1"].isin(ONSET_STATES)]
    shared = onset.groupby(
        ["session", "checkpoint", "market_proxy_v1", "shock_sign_v1"],
        sort=False,
    )["market_shock_event_id_v1"].nunique()
    if (shared != 1).any():
        raise ExperimentBlocked("stocks at one shock do not share one event ID")
    if thresholds[6].calibration_complete_v1:
        raise ExperimentBlocked("checkpoint 6 unexpectedly crossed the session")
    if not all(
        thresholds[checkpoint].calibration_complete_v1
        for checkpoint in FROZEN_SHOCK_CHECKPOINTS_V1
        if checkpoint != 6
    ):
        raise ExperimentBlocked("supported post-open checkpoint calibration incomplete")
    partition = episodes.loc[
        episodes["primary_outcome_complete_v1"].astype(bool),
        "primary_outcome_state_v1",
    ]
    if not partition.isin(
        ["MATERIAL_UP", "MATERIAL_DOWN", "NO_MATERIAL_MOVE"]
    ).all():
        raise ExperimentBlocked("target partition is not exhaustive")
    for name, draws in (
        ("session bootstrap", session_bootstrap),
        ("event bootstrap", event_bootstrap),
    ):
        counts = draws.groupby("period")["draw"].nunique()
        if set(counts.index) != {"assessment", "stress"} or not counts.eq(
            BOOTSTRAP_DRAWS
        ).all():
            raise ExperimentBlocked(f"{name} replication count drifted")
    null_counts = null_draws.groupby("period")["draw"].nunique()
    if set(null_counts.index) != {"assessment", "stress"} or not null_counts.eq(
        NULL_DRAWS
    ).all():
        raise ExperimentBlocked("null replication count drifted")
    prohibited_columns = {
        "archived_signed_pressure",
        "archived_tension",
        "option_pnl",
        "option_return",
    }
    if prohibited_columns.intersection(episodes.columns):
        raise ExperimentBlocked("prohibited field entered episode analysis")


def run() -> dict[str, Any]:
    PRIMARY.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    contract_bytes = CONTRACT_PATH.read_bytes()
    configuration_hash = hashlib.sha256(contract_bytes).hexdigest()
    source_records = _verify_sources()
    episodes_source, checkpoints_source, bars, options, vti_raw = _load_inputs()
    vti_bars, bounded_vti_hash = _prepare_vti_bars(vti_raw)
    alignment_audit = _proxy_alignment_audit(vti_bars, bars)
    market_windows = _build_market_windows(vti_bars)
    thresholds = freeze_checkpoint_thresholds_v1(market_windows)
    market_states = _apply_market_states(market_windows, thresholds)
    proxy_audit = _market_proxy_audit(
        alignment=alignment_audit,
        vti=vti_raw,
        market_states=market_states,
        bounded_hash=bounded_vti_hash,
    )

    episodes = _prepare_panel(
        episodes_source,
        population="fresh_high_m1c_episode",
        bars=bars,
        options=options,
        market_states=market_states,
        attach_responses=True,
    )
    regression = _frozen_regression(episodes_source, episodes)
    high_tail_source = checkpoints_source.loc[
        checkpoints_source["m1c_high_tail_v1"].astype(bool)
    ].copy()
    high_tail = _prepare_panel(
        high_tail_source,
        population="all_high_m1c_checkpoint_rows_tail_diagnostic",
        bars=bars,
        options=options,
        market_states=market_states,
        attach_responses=True,
    )
    quintiles = freeze_response_quintiles_v1(episodes)
    episodes["response_quintile_v1"] = [
        assign_response_quintile_v1(
            None if pd.isna(value) else float(value),
            quintiles,
        )
        for value in episodes["shock_relative_response_v1"]
    ]
    high_tail["response_quintile_v1"] = [
        assign_response_quintile_v1(
            None if pd.isna(value) else float(value),
            quintiles,
        )
        for value in high_tail["shock_relative_response_v1"]
    ]
    donors = _prepare_panel(
        checkpoints_source,
        population="all_causal_checkpoint_null_donor",
        bars=bars,
        options=options,
        market_states=market_states,
        attach_responses=False,
    )

    events = _unique_shock_events(market_states, episodes)
    accounting = _event_accounting(market_states, episodes, events)
    calibration = _calibration_table(thresholds)
    assessment_outcomes = _required_outcome_table(
        episodes,
        period="assessment",
    )
    stress_outcomes = _required_outcome_table(episodes, period="stress")
    continuation = pd.concat(
        [
            _arm_results(episodes, period=period, arm="continuation")
            for period in ("assessment", "stress")
        ],
        ignore_index=True,
    )
    resistance = pd.concat(
        [
            _arm_results(episodes, period=period, arm="resistance")
            for period in ("assessment", "stress")
        ],
        ignore_index=True,
    )
    checkpoint_strata = _checkpoint_stratified_mechanism_results(episodes)
    ranking_parts = [
        _ranking_tables(episodes, period=period)
        for period in ("assessment", "stress")
    ]
    ranking = pd.concat([part[0] for part in ranking_parts], ignore_index=True)
    quintile_outcomes = pd.concat(
        [part[1] for part in ranking_parts],
        ignore_index=True,
    )
    baselines = pd.concat(
        [
            _baseline_tables(episodes, period=period)
            for period in ("assessment", "stress")
        ],
        ignore_index=True,
    )
    normal = _normal_regime_comparison(episodes)
    sign_results = _shock_sign_stratification(episodes)
    tail = _tail_phase_diagnostics(high_tail)
    missingness = _missingness_table(episodes)
    concentration = _concentration_report(episodes)

    session_bootstrap = pd.concat(
        [
            _cluster_bootstrap(
                episodes,
                period=period,
                cluster_column="session",
                draws=BOOTSTRAP_DRAWS,
                seed=SESSION_BOOTSTRAP_SEED,
            )
            for period in ("assessment", "stress")
        ],
        ignore_index=True,
    )
    event_bootstrap = pd.concat(
        [
            _cluster_bootstrap(
                episodes,
                period=period,
                cluster_column="market_shock_event_id_v1",
                draws=BOOTSTRAP_DRAWS,
                seed=EVENT_BOOTSTRAP_SEED,
            )
            for period in ("assessment", "stress")
        ],
        ignore_index=True,
    )
    bootstrap_summary = _bootstrap_summary(
        pd.concat([session_bootstrap, event_bootstrap], ignore_index=True)
    )
    _enrich_primary_results(
        continuation=continuation,
        resistance=resistance,
        ranking=ranking,
        baselines=baselines,
        bootstrap_summary=bootstrap_summary,
    )

    null_parts: list[pd.DataFrame] = []
    null_audits: list[dict[str, Any]] = []
    for period in ("assessment", "stress"):
        null_frame, null_audit = _primary_null(
            episodes,
            donors,
            period=period,
            draws=NULL_DRAWS,
            seed=NULL_SEED,
        )
        null_parts.append(null_frame)
        null_audits.append(null_audit)
    null_draws = pd.concat(null_parts, ignore_index=True)
    temporal_placebo = [
        _temporal_placebo(episodes, period=period)
        for period in ("assessment", "stress")
    ]
    observed_assessment = {
        "continuation": float(
            continuation.loc[
                continuation["period"].eq("assessment"),
                "mean_aligned_return",
            ].iloc[0]
        ),
        "resistance": float(
            resistance.loc[
                resistance["period"].eq("assessment"),
                "mean_aligned_return",
            ].iloc[0]
        ),
    }
    assessment_null = null_draws.loc[null_draws["period"].eq("assessment")]
    raw_p = {
        "continuation": _one_sided_null_p(
            assessment_null["continuation_mean_aligned_return"],
            observed_assessment["continuation"],
        ),
        "resistance": _one_sided_null_p(
            assessment_null["resistance_mean_aligned_return"],
            observed_assessment["resistance"],
        ),
    }
    holm = _holm_two(raw_p)
    null_summary = {
        "primary_null_audit": null_audits,
        "assessment_observed": observed_assessment,
        "assessment_one_sided_raw_p_values": raw_p,
        "assessment_holm_adjusted_p_values": holm,
        "temporal_placebo": temporal_placebo,
    }

    leave_month = _leave_one_out(episodes, omitted_dimension="month_v1")
    leave_stock = _leave_one_out(episodes, omitted_dimension="stock")
    leave_event = _leave_one_out(
        episodes,
        omitted_dimension="market_shock_event_id_v1",
    )
    leave_checkpoint = _leave_one_out(episodes, omitted_dimension="checkpoint")
    leave_session = _leave_one_out(episodes, omitted_dimension="session")
    leave_tables = [
        leave_month,
        leave_stock,
        leave_event,
        leave_checkpoint,
        leave_session,
    ]
    decisions = _decision_contract(
        continuation=continuation,
        resistance=resistance,
        ranking=ranking,
        baselines=baselines,
        sign_results=sign_results,
        bootstrap_summary=bootstrap_summary,
        holm=holm,
        leave_tables=leave_tables,
        normal_comparison=normal,
    )
    continuation["decision"] = decisions["continuation_decision"]
    resistance["decision"] = decisions["resistance_decision"]

    _validate_before_write(
        episodes=episodes,
        market_states=market_states,
        events=events,
        thresholds=thresholds,
        session_bootstrap=session_bootstrap,
        event_bootstrap=event_bootstrap,
        null_draws=null_draws,
    )

    threshold_manifest = _threshold_manifest(
        thresholds,
        configuration_hash=configuration_hash,
    )
    quintile_manifest = {
        "schema_version": "m1c-signed-market-shock-response-quintiles-v1",
        "calibration": quintiles.model_dump(mode="json"),
        "calibration_population": (
            "2024 valid fresh high-M1C signed-shock-onset predictor rows only"
        ),
        "freshness_required_for_quintile_calibration": True,
        "minimum_support_not_invented": True,
        "outcomes_used": False,
        "configuration_hash": configuration_hash,
    }
    configuration = cast(
        dict[str, Any],
        json.loads(CONTRACT_PATH.read_text(encoding="utf-8")),
    )
    configuration["configuration_sha256"] = configuration_hash
    episode_artifact = episodes.loc[:, _selected_episode_columns(episodes)].copy()
    target_partition_audit = {
        "valid_finite_row_count": int(
            episodes["primary_outcome_complete_v1"].astype(bool).sum()
        ),
        "directional_union_equals_frozen_strict_event": True,
        "positive_threshold_equality_state": "NO_MATERIAL_MOVE",
        "negative_threshold_equality_state": "NO_MATERIAL_MOVE",
        "stored_m1c_target_changed": False,
    }
    summary = {
        "research_id": "M1C Signed Market Shock Transition V1",
        "schema_version": "m1c-signed-market-shock-transition-v1",
        "canonical_market_proxy": MARKET_SHOCK_PROXY_V1,
        "configuration_hash": configuration_hash,
        "chronology": configuration["chronology"],
        "event_accounting": accounting.to_dict(orient="records"),
        "continuation_results": continuation.to_dict(orient="records"),
        "resistance_results": resistance.to_dict(orient="records"),
        "continuous_ranking_results": ranking.to_dict(orient="records"),
        "bootstrap_summary": bootstrap_summary.to_dict(orient="records"),
        "null_and_placebo": null_summary,
        "decisions": decisions,
        "frozen_regression": regression,
        "target_partition_audit": target_partition_audit,
        "confirmations": {
            "m1c_unchanged": True,
            "tail_phase_v1_unchanged": True,
            "a1_unchanged": True,
            "contaminated_fields_used": False,
            "protected_2026_outcomes_accessed": False,
            "broker_accessed": False,
            "order_routing_enabled": False,
            "order_placed": False,
            "option_profitability_tested": False,
        },
    }
    payloads: dict[str, tuple[str, Any]] = {
        "frozen_configuration_v1.json": ("json", configuration),
        "canonical_market_proxy_audit_v1.json": ("json", proxy_audit),
        "checkpoint_shock_threshold_manifest_v1.json": (
            "json",
            threshold_manifest,
        ),
        "predictor_calibration_table_v1.csv": ("csv", calibration),
        "response_quintile_manifest_v1.json": ("json", quintile_manifest),
        "market_state_surface_v1.parquet": ("parquet", market_states),
        "episode_market_state_response_v1.parquet": (
            "parquet",
            episode_artifact,
        ),
        "unique_shock_events_v1.csv": ("csv", events),
        "event_accounting_v1.csv": ("csv", accounting),
        "missingness_reasons_v1.csv": ("csv", missingness),
        "target_partition_audit_v1.json": ("json", target_partition_audit),
        "assessment_outcomes_v1.csv": ("csv", assessment_outcomes),
        "stress_outcomes_v1.csv": ("csv", stress_outcomes),
        "continuation_arm_results_v1.csv": ("csv", continuation),
        "resistance_arm_results_v1.csv": ("csv", resistance),
        "checkpoint_stratified_mechanism_results_v1.csv": (
            "csv",
            checkpoint_strata,
        ),
        "continuous_ranking_results_v1.csv": ("csv", ranking),
        "response_quintile_outcomes_v1.csv": ("csv", quintile_outcomes),
        "shock_sign_stratification_v1.csv": ("csv", sign_results),
        "normal_regime_comparison_v1.csv": ("csv", normal),
        "baseline_comparisons_v1.csv": ("csv", baselines),
        "tail_phase_diagnostics_v1.csv": ("csv", tail),
        "session_cluster_bootstrap_v1.parquet": (
            "parquet",
            session_bootstrap,
        ),
        "shock_event_cluster_bootstrap_v1.parquet": (
            "parquet",
            event_bootstrap,
        ),
        "cluster_bootstrap_summary_v1.csv": ("csv", bootstrap_summary),
        "leave_one_month_out_v1.csv": ("csv", leave_month),
        "leave_one_stock_out_v1.csv": ("csv", leave_stock),
        "leave_one_shock_event_out_v1.csv": ("csv", leave_event),
        "leave_one_checkpoint_out_v1.csv": ("csv", leave_checkpoint),
        "leave_one_session_out_v1.csv": ("csv", leave_session),
        "primary_null_draws_v1.parquet": ("parquet", null_draws),
        "null_and_placebo_results_v1.json": ("json", null_summary),
        "concentration_report_v1.csv": ("csv", concentration),
        "summary_v1.json": ("json", summary),
    }
    report_path = REPORTS / "m1c_signed_market_shock_transition_v1.md"
    provenance_path = PRIMARY / "provenance_manifest_v1.json"
    summary["artifact_paths"] = {
        name: str(PRIMARY / name) for name in payloads
    } | {"report": str(report_path), "provenance": str(provenance_path)}
    artifact_hashes = _write_artifacts(payloads=payloads)

    report = _report_markdown(
        proxy_audit=proxy_audit,
        calibration=calibration,
        accounting=accounting,
        assessment_outcomes=assessment_outcomes,
        stress_outcomes=stress_outcomes,
        continuation=continuation,
        resistance=resistance,
        ranking=ranking,
        baselines=baselines,
        normal=normal,
        sign_results=sign_results,
        decisions=decisions,
        null_summary=null_summary,
        tail=tail,
        missingness=missingness,
    )
    report_path.write_text(report, encoding="utf-8")
    artifact_hashes[str(report_path.relative_to(EXPERIMENT_DIR))] = _sha256_file(
        report_path
    )

    branch = _git("symbolic-ref", "--short", "HEAD")
    commit = _git("rev-parse", "HEAD")
    dirty = _git("status", "--short")
    source_records.append(
        {
            "path": str(VTI_PATH),
            "sha256": bounded_vti_hash,
            "hash_scope": (
                "Arrow hash of predicate-materialised rows from 2024-01-01 through "
                "2025-12-31 only; protected rows were not materialised"
            ),
            "row_count": int(len(vti_raw)),
        }
    )
    provenance = {
        "schema_version": "m1c-signed-market-shock-transition-provenance-v1",
        "generated_at_utc": datetime.now(UTC),
        "repository": {
            "path": str(REPO_ROOT),
            "branch": branch,
            "commit": commit,
            "dirty_working_tree": bool(dirty),
            "dirty_status": dirty.splitlines(),
            "local_repository_authoritative": True,
        },
        "input_artifacts": source_records,
        "market_proxy": proxy_audit,
        "data_date_boundaries": configuration["chronology"],
        "calibration_row_counts_by_checkpoint": calibration[
            [
                "checkpoint",
                "market_return_w0_support_v1",
                "market_range_w0_support_v1",
                "market_return_w1_support_v1",
                "market_range_w1_support_v1",
                "calibration_complete_v1",
                "calibration_missing_reason_v1",
            ]
        ].to_dict(orient="records"),
        "episode_counts": {
            period: int(episodes["partition"].eq(period).sum())
            for period in ("development", "assessment", "stress")
        },
        "unique_shock_event_counts": {
            period: int(events["partition"].eq(period).sum())
            for period in ("development", "assessment", "stress")
        },
        "missingness_counts_and_reasons": missingness.to_dict(orient="records"),
        "configuration_sha256": configuration_hash,
        "exact_commands": [
            RUN_COMMAND,
            FOCUSED_TEST_COMMAND,
            TARGETED_RUFF_COMMAND,
            TARGETED_MYPY_COMMAND,
            "rtk uv run pytest -q",
            "rtk uv run ruff check .",
        ],
        "random_seeds": {
            "session_cluster_bootstrap": SESSION_BOOTSTRAP_SEED,
            "shock_event_cluster_bootstrap": EVENT_BOOTSTRAP_SEED,
            "primary_null": NULL_SEED,
        },
        "replications": {
            "session_cluster_bootstrap": BOOTSTRAP_DRAWS,
            "shock_event_cluster_bootstrap": BOOTSTRAP_DRAWS,
            "primary_null": NULL_DRAWS,
        },
        "frozen_regression": regression,
        "target_partition_audit": target_partition_audit,
        "output_sha256": artifact_hashes,
        "confirmations": summary["confirmations"],
        "causality_confirmations": {
            "market_windows_end_no_later_than_signal": True,
            "entry_bar_excluded": True,
            "future_market_bars_used": False,
            "future_stock_bars_used_for_response": False,
            "future_option_observations_used": False,
            "checkpoint_pooling_used": False,
            "outcome_driven_thresholds_used": False,
            "cross_sectional_normalisation_used": False,
            "stock_or_month_predictive_inputs_used": False,
        },
        "execution_confirmations": {
            "broker_access": False,
            "order_routing_enabled": False,
            "orders_submitted": False,
        },
        "protected_data_confirmation": {
            "protected_data_opened": False,
            "protected_outcomes_calculated": False,
            "protected_outcomes_displayed": False,
            "protected_outcomes_inspected": False,
            "bounded_reader_upper_limit_exclusive": PROTECTED_START,
        },
    }
    _write_json(provenance_path, provenance)
    return summary


def main() -> int:
    try:
        summary = run()
    except ExperimentBlocked as exc:
        print(f"operational_failure: {exc}", file=sys.stderr)
        return 2
    decisions = cast(dict[str, Any], summary["decisions"])
    print(json.dumps(decisions, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
