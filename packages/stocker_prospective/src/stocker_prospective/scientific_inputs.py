"""Build causal pre-session inputs for the frozen M1C recorder."""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from collections import Counter
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from threading import Event
from typing import Any, cast
from zoneinfo import ZoneInfo

import httpx
import pandas as pd

from stocker_core.config import EODHDConfig
from stocker_data.vendors.eodhd import (
    EODHDClient,
    normalize_eod_response,
    normalize_intraday_response,
)
from stocker_prospective.append_only import write_immutable_bytes, write_immutable_json
from stocker_prospective.context import previous_xnys_session
from stocker_prospective.group_o import (
    GROUP_O_FEATURE_MANIFEST_SHA256,
    GROUP_O_REGIME_MAPPING_SHA256,
    FrozenGroupOContext,
    FrozenGroupOSessionPackage,
    GroupORevisionCutoffError,
    append_group_o_session_revision,
    build_group_o_context,
    load_group_o_session_package,
    requires_group_o_implied_movement_correction,
)
from stocker_prospective.live_bars import xnys_session_bounds
from stocker_research.broad_conflict_options_iv_screen_v0 import (
    calculate_primary_option_features,
    select_primary_atm_pair,
)
from stocker_research.daily_soft_regimes_v0 import (
    FrozenDimensionParameters,
    RobustValueScale,
)
from stocker_research.daily_stock_options_context_v0 import (
    calculate_daily_stock_raw_features,
)
from stocker_research.eodhd_options_downloader_v0 import (
    DownloadConfig,
    EODHDOptionsDownloader,
    OptionsRequest,
    ProviderDTEPolicy,
    TransportLike,
    canonicalize_response_records,
    resolve_canonical_duplicates,
)
from stocker_research.front_options_soft_regimes_v01 import (
    FRONT_OPTIONS_DIMENSIONS,
    apply_front_options_dimensions,
    apply_serialized_diag_regime,
)
from stocker_research.m1c_low_movement_v0 import iv_expected_absolute

GROUP_O_CONTRACT_VERSION = "frozen-m1c-microstructure-recorder-v0/group-o-session-v0"
NEW_YORK = ZoneInfo("America/New_York")
GROUP_O_MISSING_INDICATORS = (
    "skew_25d_missing",
    "near_spot_oi_concentration_missing",
    "call_put_oi_imbalance_missing",
)
GROUP_O_REGIME_FEATURES = (
    "front_options_regime_p_0",
    "front_options_regime_p_1",
    "front_options_regime_p_2",
    "front_options_regime_p_3",
    "front_options_regime_entropy",
    "front_options_regime_margin",
)
GROUP_O_FEATURES = (
    *FRONT_OPTIONS_DIMENSIONS,
    *GROUP_O_REGIME_FEATURES,
    *GROUP_O_MISSING_INDICATORS,
)
GROUP_O_RETRY_DELAY = timedelta(minutes=15)


@dataclass(frozen=True)
class HistoricalActivityBuildResult:
    output_path: Path
    row_count: int
    symbol_count: int
    session_count: int
    latest_authorised_session: date


@dataclass(frozen=True)
class GroupOAcquisitionResult:
    output_path: Path
    signal_session: date
    observation_session: date
    symbol_count: int
    canonical_option_rows: int
    rejected_option_rows: int


class GroupOAcquisitionPending(RuntimeError):
    """The exact D-1 chain is not published yet and acquisition must retry."""


def allocate_group_o_attempt(
    *,
    cache_root: str | Path,
    observation_session: date,
) -> tuple[str, Path]:
    """Reserve the next append-only attempt directory without an allocation race."""

    attempts_root = Path(cache_root) / observation_session.isoformat() / "attempts"
    attempts_root.mkdir(parents=True, exist_ok=True)
    existing = [
        int(path.name)
        for path in attempts_root.iterdir()
        if path.is_dir() and len(path.name) == 4 and path.name.isdigit()
    ]
    for number in range(max(existing, default=0) + 1, 10_000):
        attempt_id = f"{number:04d}"
        attempt_path = attempts_root / attempt_id
        try:
            attempt_path.mkdir()
        except FileExistsError:
            continue
        return attempt_id, attempt_path
    raise RuntimeError("Group O acquisition attempt identity exhausted")


@dataclass(frozen=True)
class FrozenGroupOArtifacts:
    parameters: FrozenDimensionParameters
    regime_mapping: Mapping[str, Any]
    feature_manifest_hash: str
    regime_mapping_hash: str

    @classmethod
    def load(
        cls,
        *,
        feature_manifest_path: str | Path,
        regime_mapping_path: str | Path,
    ) -> FrozenGroupOArtifacts:
        feature_path = Path(feature_manifest_path)
        regime_path = Path(regime_mapping_path)
        feature_hash = _sha256_path(feature_path)
        regime_hash = _sha256_path(regime_path)
        if feature_hash != GROUP_O_FEATURE_MANIFEST_SHA256:
            raise ValueError("frozen Group O feature manifest hash differs")
        if regime_hash != GROUP_O_REGIME_MAPPING_SHA256:
            raise ValueError("frozen Group O regime mapping hash differs")
        return cls(
            parameters=_dimension_parameters(feature_path),
            regime_mapping=cast(
                dict[str, Any],
                json.loads(regime_path.read_text(encoding="utf-8")),
            ),
            feature_manifest_hash=feature_hash,
            regime_mapping_hash=regime_hash,
        )


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _xnys_schedule(start: date, end: date) -> pd.DataFrame:
    import pandas_market_calendars as mcal

    return cast(
        pd.DataFrame,
        mcal.get_calendar("XNYS").schedule(start_date=start, end_date=end),
    )


def build_historical_activity_baseline(
    *,
    source_frames: Mapping[str, pd.DataFrame],
    latest_authorised_session: date,
    output_path: str | Path,
    minimum_sessions: int = 10,
) -> HistoricalActivityBuildResult:
    """Materialize one prior-session-only regular-hours volume stream per stock."""

    if not source_frames:
        raise ValueError("historical activity source frames are empty")
    if minimum_sessions <= 0:
        raise ValueError("historical activity minimum sessions must be positive")
    normalized: dict[str, pd.DataFrame] = {}
    minimum_day: date | None = None
    for symbol, source in sorted(source_frames.items()):
        missing = {"timestamp", "volume"}.difference(source.columns)
        if missing:
            raise ValueError(f"historical activity source missing columns for {symbol}: {missing}")
        frame = source.loc[:, ["timestamp", "volume"]].copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise", utc=True)
        frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce")
        if frame["timestamp"].duplicated().any():
            raise ValueError(f"historical activity source contains duplicate bars for {symbol}")
        if frame.empty:
            raise ValueError(f"historical activity source is empty for {symbol}")
        first_day = cast(pd.Timestamp, frame["timestamp"].min()).date()
        minimum_day = first_day if minimum_day is None else min(minimum_day, first_day)
        normalized[symbol] = frame
    assert minimum_day is not None
    schedule = _xnys_schedule(minimum_day, latest_authorised_session)
    rows: list[dict[str, object]] = []
    for symbol, frame in normalized.items():
        for index, schedule_row in schedule.iterrows():
            session = date.fromisoformat(str(index)[:10])
            opened = pd.Timestamp(schedule_row["market_open"]).tz_convert(UTC)
            closed = pd.Timestamp(schedule_row["market_close"]).tz_convert(UTC)
            observed = frame.loc[
                frame["timestamp"].ge(opened) & frame["timestamp"].lt(closed)
            ].sort_values("timestamp", kind="mergesort")
            finite = observed["volume"].map(lambda value: math.isfinite(float(value)))
            observed = observed.loc[finite & observed["volume"].ge(0.0)]
            for timestamp_value, volume_value in zip(
                observed["timestamp"],
                observed["volume"],
                strict=True,
            ):
                timestamp = pd.Timestamp(cast(Any, timestamp_value))
                offset_seconds = (timestamp - opened).total_seconds()
                if offset_seconds < 0 or offset_seconds % 300 != 0:
                    raise ValueError(
                        f"historical activity bar boundary differs for {symbol} {session}"
                    )
                rows.append(
                    {
                        "symbol": symbol,
                        "session": session.isoformat(),
                        "bar_ordinal": int(offset_seconds // 300),
                        "volume": float(cast(Any, volume_value)),
                    }
                )
    output = pd.DataFrame(rows, columns=["symbol", "session", "bar_ordinal", "volume"])
    if output.empty:
        raise ValueError("historical activity baseline has no regular-session rows")
    if output.duplicated(["symbol", "session", "bar_ordinal"]).any():
        raise ValueError("historical activity baseline contains duplicate identities")
    support = (
        output.loc[output["bar_ordinal"].between(0, 33)]
        .groupby(["symbol", "bar_ordinal"], sort=True)["session"]
        .nunique()
    )
    for symbol in source_frames:
        for ordinal in range(34):
            if int(support.get((symbol, ordinal), 0)) < minimum_sessions:
                raise ValueError(
                    f"historical activity baseline lacks {minimum_sessions} sessions "
                    f"for {symbol} ordinal {ordinal}"
                )
    output = output.sort_values(
        ["symbol", "session", "bar_ordinal"],
        kind="mergesort",
    ).reset_index(drop=True)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise ValueError("historical activity baseline destination cannot be a symlink")
    if destination.is_file():
        existing = pd.read_parquet(destination)
        if not existing.equals(output):
            raise ValueError("immutable historical activity baseline differs")
        return HistoricalActivityBuildResult(
            output_path=destination,
            row_count=len(output),
            symbol_count=int(output["symbol"].nunique()),
            session_count=int(output[["symbol", "session"]].drop_duplicates().shape[0]),
            latest_authorised_session=latest_authorised_session,
        )
    if destination.exists():
        raise ValueError("historical activity baseline destination is not a regular file")
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        output.to_parquet(temporary, index=False)
        os.link(temporary, destination)
    except FileExistsError:
        existing = pd.read_parquet(destination)
        if not existing.equals(output):
            raise ValueError("immutable historical activity baseline differs") from None
    finally:
        temporary.unlink(missing_ok=True)
    return HistoricalActivityBuildResult(
        output_path=destination,
        row_count=len(output),
        symbol_count=int(output["symbol"].nunique()),
        session_count=int(output[["symbol", "session"]].drop_duplicates().shape[0]),
        latest_authorised_session=latest_authorised_session,
    )


def acquire_eodhd_historical_activity_baseline(
    *,
    symbols: tuple[str, ...],
    from_session: date,
    latest_authorised_session: date,
    output_path: str | Path,
    minimum_sessions: int = 10,
    heartbeat: Callable[[], object] | None = None,
) -> HistoricalActivityBuildResult:
    """Acquire the explicit frozen EODHD interval and create the baseline once."""

    if from_session > latest_authorised_session:
        raise ValueError("historical activity acquisition range is inverted")
    if len(symbols) != 20 or len(set(symbols)) != 20:
        raise ValueError("historical activity acquisition requires the exact 20-stock cohort")
    client = EODHDClient(
        config=EODHDConfig(
            request_timeout_seconds=30.0,
            max_retries=3,
        )
    )
    client.require_token()
    frames: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        if heartbeat is not None:
            heartbeat()
        payload = client.fetch_intraday(
            symbol=f"{symbol}.US",
            from_date=from_session.isoformat(),
            to_date=latest_authorised_session.isoformat(),
            interval="5m",
        )
        frame = normalize_intraday_response(
            payload,
            symbol=symbol,
            instrument_type="equity",
            interval="5m",
        )
        frames[symbol] = frame.loc[:, ["timestamp", "volume"]]
    return build_historical_activity_baseline(
        source_frames=frames,
        latest_authorised_session=latest_authorised_session,
        output_path=output_path,
        minimum_sessions=minimum_sessions,
    )


def _finite(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(cast(Any, value))
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _nearest_delta(
    frame: pd.DataFrame,
    *,
    option_type: str,
    target: float,
) -> Mapping[str, Any] | None:
    candidates: list[tuple[tuple[float, str], Mapping[str, Any]]] = []
    for row in cast(list[dict[str, Any]], frame.to_dict(orient="records")):
        if str(row.get("option_type", "")).casefold() != option_type:
            continue
        delta = _finite(row.get("delta"))
        implied_volatility = _finite(row.get("implied_volatility"))
        if delta is None or implied_volatility is None or implied_volatility <= 0.0:
            continue
        error = abs(delta - target)
        if error <= 0.10:
            candidates.append(((error, str(row.get("contract_id", ""))), row))
    return None if not candidates else min(candidates, key=lambda value: value[0])[1]


def _prospective_surface(
    chain: pd.DataFrame,
    *,
    observation_session: date,
    previous_close: float,
    realised_volatility_20d: float,
) -> dict[str, object]:
    if chain.empty:
        return {"pair_available": False, "pair_reason": "missing_exact_chain"}
    working = chain.copy()
    working["trade_date"] = pd.to_datetime(working["trade_date"], errors="raise").dt.date
    observed_dates = tuple(sorted(set(working["trade_date"])))
    if observed_dates != (observation_session,):
        raise ValueError("prospective Group O chain must contain the exact D-1 session")
    working["expiration_date"] = pd.to_datetime(
        working["expiration_date"],
        errors="coerce",
    ).dt.date
    working["dte"] = working["expiration_date"].map(
        lambda value: (value - observation_session).days if isinstance(value, date) else math.nan
    )
    front = select_primary_atm_pair(working, previous_close=previous_close)
    if not front.available:
        return {
            "pair_available": False,
            "pair_reason": front.reason,
            "front_expiration_date": front.expiration_date,
            "front_dte": front.dte,
            "front_strike": front.strike,
        }
    if front.expiration_date is None:
        raise AssertionError("available Group O front pair lacks expiration")
    primary = calculate_primary_option_features(front, previous_close=previous_close)
    front_chain = working.loc[working["expiration_date"].eq(front.expiration_date)].copy()
    put_25 = _nearest_delta(front_chain, option_type="put", target=-0.25)
    call_25 = _nearest_delta(front_chain, option_type="call", target=0.25)
    put_iv = None if put_25 is None else _finite(put_25.get("implied_volatility"))
    call_iv = None if call_25 is None else _finite(call_25.get("implied_volatility"))
    skew = None if put_iv is None or call_iv is None else put_iv - call_iv

    open_interest = pd.to_numeric(front_chain["open_interest"], errors="coerce")
    valid_oi = open_interest.notna() & open_interest.ge(0.0)
    valid_front = front_chain.loc[valid_oi].copy()
    valid_front["_valid_open_interest"] = open_interest.loc[valid_oi].to_numpy(float)
    total_oi = float(valid_front["_valid_open_interest"].sum())
    concentration: float | None = None
    imbalance: float | None = None
    if total_oi > 0.0:
        near = valid_front["strike"].between(
            previous_close * 0.95,
            previous_close * 1.05,
            inclusive="both",
        )
        concentration = float(valid_front.loc[near, "_valid_open_interest"].sum()) / total_oi
        types = valid_front["option_type"].astype(str).str.casefold()
        call_oi = float(valid_front.loc[types.eq("call"), "_valid_open_interest"].sum())
        put_oi = float(valid_front.loc[types.eq("put"), "_valid_open_interest"].sum())
        imbalance = math.log((call_oi + 1.0) / (put_oi + 1.0))
    if not math.isfinite(realised_volatility_20d) or realised_volatility_20d < 0.0:
        raise ValueError("prospective Group O realised volatility is invalid")
    return {
        "pair_available": True,
        "pair_reason": "selected",
        "front_expiration_date": front.expiration_date,
        "front_dte": front.dte,
        "front_strike": front.strike,
        "atm_iv": float(primary["atm_iv"]),
        "straddle_mid_pct": float(primary["straddle_mid_pct"]),
        "call_put_iv_gap": float(primary["call_put_iv_gap"]),
        "skew_25d": skew,
        "combined_relative_spread": float(primary["combined_relative_spread"]),
        "iv_minus_realised_20d": float(primary["atm_iv"]) - realised_volatility_20d,
        "near_spot_oi_concentration": concentration,
        "call_put_oi_imbalance": imbalance,
        "skew_25d_missing": int(skew is None),
        "near_spot_oi_concentration_missing": int(concentration is None),
        "call_put_oi_imbalance_missing": int(imbalance is None),
    }


def _dimension_parameters(path: str | Path) -> FrozenDimensionParameters:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return FrozenDimensionParameters(
        kind=str(payload["kind"]),
        scales={
            str(name): RobustValueScale(
                center=float(cast(Any, values["center"])),
                scale=float(cast(Any, values["scale"])),
            )
            for name, values in cast(dict[str, dict[str, object]], payload["scales"]).items()
        },
        imputation_medians={
            str(name): float(cast(Any, value))
            for name, value in cast(
                dict[str, object],
                payload["imputation_medians"],
            ).items()
        },
        fitted_period=str(payload["fitted_period"]),
    )


def _context_features(
    surface: Mapping[str, object],
    *,
    parameters: FrozenDimensionParameters,
    regime_mapping: Mapping[str, Any],
) -> dict[str, float | int | bool | None]:
    raw = pd.DataFrame([dict(surface)])
    dimensions = apply_front_options_dimensions(raw, parameters)
    regime = apply_serialized_diag_regime(
        dimensions,
        regime_mapping,
        prefix="front_options_regime",
    )
    row = regime.iloc[0]
    result: dict[str, float | int | bool | None] = {}
    for name in GROUP_O_FEATURES:
        value = row[name]
        if name in GROUP_O_MISSING_INDICATORS:
            result[name] = int(value)
        else:
            result[name] = float(value)
    return result


def build_group_o_session_package(
    *,
    signal_session: date,
    symbols: tuple[str, ...],
    canonical_options_by_symbol: Mapping[str, pd.DataFrame],
    daily_bars: pd.DataFrame,
    source_receipt_hashes_by_symbol: Mapping[str, tuple[str, ...]],
    feature_manifest_path: str | Path,
    regime_mapping_path: str | Path,
    output_path: str | Path,
    implied_movement_atm_iv_by_symbol_out: dict[str, float] | None = None,
) -> FrozenGroupOSessionPackage:
    """Build exact D-1 Group O contexts with frozen 2024 transforms only."""

    if not symbols or len(symbols) != len(set(symbols)):
        raise ValueError("Group O symbols must be non-empty and unique")
    artifacts = FrozenGroupOArtifacts.load(
        feature_manifest_path=feature_manifest_path,
        regime_mapping_path=regime_mapping_path,
    )
    observation_session = previous_xnys_session(signal_session)
    required_daily = {"symbol", "session", "open", "high", "low", "close", "activity"}
    if missing := required_daily.difference(daily_bars.columns):
        raise ValueError(f"Group O daily bars missing columns: {sorted(missing)}")
    stock_features = calculate_daily_stock_raw_features(daily_bars)
    contexts: list[FrozenGroupOContext] = []
    for symbol in symbols:
        stock_rows = stock_features.loc[
            stock_features["symbol"].eq(symbol) & stock_features["session"].eq(observation_session)
        ]
        receipts = tuple(
            dict.fromkeys(
                (
                    *source_receipt_hashes_by_symbol.get(symbol, ()),
                    artifacts.feature_manifest_hash,
                    artifacts.regime_mapping_hash,
                )
            )
        )
        if len(stock_rows) != 1:
            surface: dict[str, object] = {
                "pair_available": False,
                "pair_reason": "missing_exact_daily_stock_context",
            }
        else:
            stock = stock_rows.iloc[0]
            surface = _prospective_surface(
                canonical_options_by_symbol.get(symbol, pd.DataFrame()),
                observation_session=observation_session,
                previous_close=float(stock["unadjusted_close"]),
                realised_volatility_20d=float(stock["realised_volatility_20d"]),
            )
        pair_available = surface.get("pair_available") is True
        atm_iv = float(cast(Any, surface["atm_iv"])) if pair_available else None
        if implied_movement_atm_iv_by_symbol_out is not None and atm_iv is not None:
            implied_movement_atm_iv_by_symbol_out[symbol] = atm_iv
        if pair_available:
            features = _context_features(
                surface,
                parameters=artifacts.parameters,
                regime_mapping=artifacts.regime_mapping,
            )
        else:
            features = {name: None for name in GROUP_O_FEATURES}
        missing_indicators = {
            name: bool(features[name]) if pair_available else True
            for name in GROUP_O_MISSING_INDICATORS
        }
        contexts.append(
            build_group_o_context(
                symbol=symbol,
                signal_session=signal_session,
                actual_option_observation_session=observation_session,
                front_expiry=cast(date | None, surface.get("front_expiration_date")),
                dte=cast(int | None, surface.get("front_dte")),
                atm_strike=cast(float | None, surface.get("front_strike")),
                previous_close_implied_movement_15m=(
                    iv_expected_absolute(atm_iv, 15) if atm_iv is not None else None
                ),
                features=features,
                missing_indicators=missing_indicators,
                quality_status=(
                    "valid" if pair_available else str(surface.get("pair_reason", "invalid"))
                ),
                source_receipt_hashes=receipts,
            )
        )
    package = FrozenGroupOSessionPackage(
        contract_version=GROUP_O_CONTRACT_VERSION,
        signal_session=signal_session,
        generated_from_authorised_cache=True,
        feature_manifest_hash=artifacts.feature_manifest_hash,
        regime_mapping_hash=artifacts.regime_mapping_hash,
        contexts=tuple(contexts),
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise ValueError("Group O package destination cannot be a symlink")
    if destination.is_file():
        existing = FrozenGroupOSessionPackage.model_validate_json(
            destination.read_text(encoding="utf-8")
        )
        if existing != package:
            raise ValueError("immutable Group O package differs")
        return existing
    if destination.exists():
        raise ValueError("Group O package destination is not a regular file")
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(package.model_dump_json(indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            existing = FrozenGroupOSessionPackage.model_validate_json(
                destination.read_text(encoding="utf-8")
            )
            if existing != package:
                raise ValueError("immutable Group O package differs") from None
            return existing
    finally:
        temporary.unlink(missing_ok=True)
    return package


def _payload_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def signed_group_o_attempt_receipt(payload: Mapping[str, object]) -> dict[str, object]:
    if "attempt_receipt_sha256" in payload:
        raise ValueError("Group O attempt receipt payload is already signed")
    signed = dict(payload)
    signed["attempt_receipt_sha256"] = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    return signed


def write_group_o_attempt_receipt(path: Path, payload: Mapping[str, object]) -> None:
    write_immutable_json(
        path,
        signed_group_o_attempt_receipt(payload),
        conflict_message="immutable Group O acquisition attempt differs",
    )


def load_group_o_attempt_receipt(path: Path) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise ValueError("Group O attempt receipt is not an immutable regular file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Group O attempt receipt is invalid") from exc
    if not isinstance(payload, dict):
        raise ValueError("Group O attempt receipt is invalid")
    unsigned = dict(payload)
    signature = unsigned.pop("attempt_receipt_sha256", None)
    expected = hashlib.sha256(
        json.dumps(
            unsigned,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    if signature != expected:
        raise ValueError("Group O attempt receipt signature differs")
    return payload


def group_o_retry_not_before(
    *,
    cache_root: str | Path,
    observation_session: date,
) -> datetime | None:
    """Return the signed persisted retry boundary for the newest completed attempt."""

    attempts_root = Path(cache_root) / observation_session.isoformat() / "attempts"
    if not attempts_root.is_dir() or attempts_root.is_symlink():
        return None
    attempts = sorted(
        (
            path
            for path in attempts_root.iterdir()
            if path.is_dir() and not path.is_symlink() and path.name.isdigit()
        ),
        key=lambda path: path.name,
        reverse=True,
    )
    for attempt in attempts:
        receipt_path = attempt / "attempt_receipt.json"
        if not receipt_path.exists():
            continue
        receipt = load_group_o_attempt_receipt(receipt_path)
        if receipt.get("status") != "pending_exact_chain":
            return None
        if (
            receipt.get("schema_version") != "group-o-acquisition-attempt-v1"
            or receipt.get("attempt_id") != attempt.name
            or len(attempt.name) != 4
            or receipt.get("observation_session") != observation_session.isoformat()
        ):
            raise ValueError("Group O pending attempt identity differs")
        raw_signal = receipt.get("signal_session")
        raw_completed = receipt.get("completed_at_utc")
        raw_retry = receipt.get("retry_after_utc")
        if not all(isinstance(value, str) for value in (raw_signal, raw_completed, raw_retry)):
            raise ValueError("Group O pending attempt lacks frozen chronology")
        try:
            signal_session = date.fromisoformat(cast(str, raw_signal))
            completed_at = datetime.fromisoformat(cast(str, raw_completed))
            retry_after = datetime.fromisoformat(cast(str, raw_retry))
        except ValueError as exc:
            raise ValueError("Group O pending attempt chronology is invalid") from exc
        if previous_xnys_session(signal_session) != observation_session:
            raise ValueError("Group O pending attempt session chronology differs")
        if (
            completed_at.tzinfo is None
            or completed_at.utcoffset() is None
            or retry_after.tzinfo is None
            or retry_after.utcoffset() is None
        ):
            raise ValueError("Group O attempt retry chronology must be timezone-aware")
        completed_at = completed_at.astimezone(UTC)
        retry_after = retry_after.astimezone(UTC)
        if retry_after != completed_at + GROUP_O_RETRY_DELAY:
            raise ValueError("Group O pending attempt retry interval differs from 15 minutes")
        return retry_after.astimezone(UTC)
    return None


def _preserve_resolved_group_o_contexts(
    *,
    previous: FrozenGroupOSessionPackage,
    candidate: FrozenGroupOSessionPackage,
    include_implied_movement_corrections: bool,
) -> FrozenGroupOSessionPackage:
    """Limit a revision to exact-chain failures and omitted derived movement."""

    previous_by_symbol = {context.symbol: context for context in previous.contexts}
    contexts: list[FrozenGroupOContext] = []
    for candidate_context in candidate.contexts:
        previous_context = previous_by_symbol[candidate_context.symbol]
        if previous_context.quality_status == "missing_exact_chain":
            contexts.append(candidate_context)
            continue
        candidate_movement = candidate_context.previous_close_implied_movement_15m
        if include_implied_movement_corrections and (
            requires_group_o_implied_movement_correction(previous_context)
        ):
            previous_pair_identity = (
                previous_context.actual_option_observation_session,
                previous_context.front_expiry,
                previous_context.dte,
                previous_context.atm_strike,
            )
            candidate_pair_identity = (
                candidate_context.actual_option_observation_session,
                candidate_context.front_expiry,
                candidate_context.dte,
                candidate_context.atm_strike,
            )
            if candidate_pair_identity != previous_pair_identity:
                raise ValueError("Group O implied-movement exact ATM pair identity differs")
            if (
                candidate_movement is None
                or not math.isfinite(candidate_movement)
                or candidate_movement <= 0.0
            ):
                contexts.append(previous_context)
                continue
            contexts.append(
                build_group_o_context(
                    symbol=previous_context.symbol,
                    signal_session=previous_context.signal_session,
                    actual_option_observation_session=(
                        previous_context.actual_option_observation_session
                    ),
                    front_expiry=previous_context.front_expiry,
                    dte=previous_context.dte,
                    atm_strike=previous_context.atm_strike,
                    previous_close_implied_movement_15m=candidate_movement,
                    features=previous_context.features,
                    missing_indicators=previous_context.missing_indicators,
                    quality_status=previous_context.quality_status,
                    source_receipt_hashes=tuple(
                        dict.fromkeys(
                            (
                                *previous_context.source_receipt_hashes,
                                *candidate_context.source_receipt_hashes,
                            )
                        )
                    ),
                )
            )
            continue
        contexts.append(previous_context)
    return FrozenGroupOSessionPackage.model_validate(
        {**candidate.model_dump(mode="python"), "contexts": tuple(contexts)}
    )


def _implied_movement_atm_iv_sources(
    package: FrozenGroupOSessionPackage,
    candidate_atm_iv_by_symbol: Mapping[str, float],
) -> dict[str, float]:
    symbols = tuple(
        context.symbol
        for context in package.contexts
        if requires_group_o_implied_movement_correction(context)
    )
    missing_sources = tuple(
        symbol for symbol in symbols if symbol not in candidate_atm_iv_by_symbol
    )
    if missing_sources:
        raise ValueError(
            "Group O implied-movement ATM IV source unavailable: " + ",".join(missing_sources)
        )
    return {symbol: candidate_atm_iv_by_symbol[symbol] for symbol in symbols}


def acquire_eodhd_group_o_session_package(
    *,
    signal_session: date,
    symbols: tuple[str, ...],
    output_path: str | Path,
    cache_root: str | Path,
    cache_attempt_id: str,
    feature_manifest_path: str | Path,
    regime_mapping_path: str | Path,
    supersedes_path: str | Path | None = None,
    provider_dte_policy: ProviderDTEPolicy = "strict_match",
    heartbeat: Callable[[], object] | None = None,
    cancellation_requested: Callable[[], bool] = lambda: False,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> GroupOAcquisitionResult:
    """Download bounded D-1 EOD evidence and materialize the frozen Group O package."""

    if len(cache_attempt_id) != 4 or not cache_attempt_id.isdigit():
        raise ValueError("Group O cache attempt ID must be four decimal digits")
    if provider_dte_policy not in {"strict_match", "recompute_from_eod_identity"}:
        raise ValueError("invalid Group O provider DTE policy")
    observation_session = previous_xnys_session(signal_session)
    signal_open, _ = xnys_session_bounds(signal_session)
    attempt_started = clock().astimezone(UTC)
    if attempt_started >= signal_open:
        raise GroupOAcquisitionPending(
            "Group O exact-chain acquisition cutoff passed before the signal session open"
        )
    eodhd = EODHDClient(
        config=EODHDConfig(
            request_timeout_seconds=10.0,
            max_retries=1,
        )
    )
    token = eodhd.require_token()
    daily_frames: list[pd.DataFrame] = []
    options_by_symbol: dict[str, pd.DataFrame] = {}
    receipt_hashes: dict[str, tuple[str, ...]] = {}
    accepted_rows_by_symbol: dict[str, int] = {}
    rejected_rows_by_symbol: dict[str, int] = {}
    provider_dte_diagnostics: list[dict[str, object]] = []
    accepted_rows = 0
    rejected_rows = 0
    cache = Path(cache_root) / observation_session.isoformat() / "attempts" / cache_attempt_id
    attempt_receipt_path = cache / "attempt_receipt.json"
    transport = httpx.Client()
    downloader = EODHDOptionsDownloader(
        DownloadConfig(
            token=token,
            data_dir=cache,
            maximum_raw_records=250_000,
            maximum_download_bytes=1_000_000_000,
            request_timeout_seconds=10.0,
            max_attempts=1,
            requests_per_minute=120,
        ),
        transport=cast(TransportLike, transport),
    )
    try:
        for symbol in symbols:
            if cancellation_requested():
                raise RuntimeError("Group O preparation cancelled")
            if heartbeat is not None:
                heartbeat()
            daily_payload = eodhd.fetch_eod(
                symbol=f"{symbol}.US",
                from_date=(observation_session - timedelta(days=75)).isoformat(),
                to_date=observation_session.isoformat(),
                period="d",
            )
            daily = normalize_eod_response(
                daily_payload,
                symbol=symbol,
                instrument_type="equity",
                period="d",
            )
            daily["symbol"] = symbol
            daily["session"] = daily["timestamp"].dt.date
            daily["activity"] = daily["volume"]
            daily_frames.append(
                daily.loc[
                    :,
                    ["symbol", "session", "open", "high", "low", "close", "activity"],
                ]
            )
            if cancellation_requested():
                raise RuntimeError("Group O preparation cancelled")
            request = OptionsRequest(
                underlying_symbol=symbol,
                trade_date_from=observation_session,
                trade_date_to=observation_session,
                expiration_from=observation_session + timedelta(days=7),
                expiration_to=observation_session + timedelta(days=45),
                compact=False,
            )
            result = downloader.download_with_splitting(request)
            canonical = canonicalize_response_records(
                result.records,
                request_id=f"group-o|{symbol}|{observation_session.isoformat()}",
                provider_schema_version="eodhd-options-eod-v1",
                provider_dte_policy=provider_dte_policy,
            )
            deduplicated = resolve_canonical_duplicates(canonical.records)
            if deduplicated.conflicting_duplicate_groups:
                raise ValueError(f"Group O options contain conflicting duplicate rows for {symbol}")
            if any(
                str(record.get("underlying_symbol", "")) != symbol
                for record in deduplicated.records
            ):
                write_group_o_attempt_receipt(
                    attempt_receipt_path,
                    {
                        "schema_version": "group-o-acquisition-attempt-v1",
                        "attempt_id": cache_attempt_id,
                        "signal_session": signal_session.isoformat(),
                        "observation_session": observation_session.isoformat(),
                        "started_at_utc": attempt_started.isoformat(),
                        "completed_at_utc": clock().astimezone(UTC).isoformat(),
                        "status": "blocked_source_underlying_identity",
                        "requested_symbol": symbol,
                    },
                )
                raise ValueError(f"Group O option underlying identity differs for {symbol}")
            if any(
                record.get("trade_date") != observation_session for record in deduplicated.records
            ):
                write_group_o_attempt_receipt(
                    attempt_receipt_path,
                    {
                        "schema_version": "group-o-acquisition-attempt-v1",
                        "attempt_id": cache_attempt_id,
                        "signal_session": signal_session.isoformat(),
                        "observation_session": observation_session.isoformat(),
                        "started_at_utc": attempt_started.isoformat(),
                        "completed_at_utc": clock().astimezone(UTC).isoformat(),
                        "status": "blocked_source_observation_identity",
                        "requested_symbol": symbol,
                    },
                )
                raise ValueError(f"Group O option observation identity differs for {symbol}")
            options_by_symbol[symbol] = pd.DataFrame(deduplicated.records)
            provider_dte_diagnostics.extend(
                diagnostic.to_dict() for diagnostic in canonical.provider_dte_diagnostics
            )
            accepted_rows_by_symbol[symbol] = len(deduplicated.records)
            rejected_rows_by_symbol[symbol] = len(canonical.rejections)
            accepted_rows += len(deduplicated.records)
            rejected_rows += len(canonical.rejections)
            receipt_hashes[symbol] = tuple(
                dict.fromkeys(
                    (
                        _payload_hash(daily_payload),
                        *(row.response_hash for row in result.manifest_rows),
                    )
                )
            )
    finally:
        transport.close()
    completed_at = clock().astimezone(UTC)
    provider_dte_diagnostics_path = cache / "provider_dte_diagnostics.json"
    provider_dte_status_counts = dict(
        sorted(Counter(str(row["status"]) for row in provider_dte_diagnostics).items())
    )
    write_immutable_json(
        provider_dte_diagnostics_path,
        {
            "schema_version": "group-o-provider-dte-diagnostics-v1",
            "attempt_id": cache_attempt_id,
            "signal_session": signal_session.isoformat(),
            "observation_session": observation_session.isoformat(),
            "provider_dte_policy": provider_dte_policy,
            "provider_dte_used_for_admission": provider_dte_policy == "strict_match",
            "diagnostic_count": len(provider_dte_diagnostics),
            "status_counts": provider_dte_status_counts,
            "rows": provider_dte_diagnostics,
        },
        conflict_message="immutable Group O provider DTE diagnostics differ",
    )
    missing_exact_chain_symbols = tuple(
        symbol for symbol in symbols if accepted_rows_by_symbol.get(symbol, 0) == 0
    )
    attempt_receipt: dict[str, object] = {
        "schema_version": "group-o-acquisition-attempt-v1",
        "attempt_id": cache_attempt_id,
        "signal_session": signal_session.isoformat(),
        "observation_session": observation_session.isoformat(),
        "started_at_utc": attempt_started.isoformat(),
        "completed_at_utc": completed_at.isoformat(),
        "symbol_count": len(symbols),
        "canonical_option_rows": accepted_rows,
        "rejected_option_rows": rejected_rows,
        "canonical_rows_by_symbol": dict(sorted(accepted_rows_by_symbol.items())),
        "rejected_rows_by_symbol": dict(sorted(rejected_rows_by_symbol.items())),
        "source_receipt_hashes_by_symbol": {
            symbol: list(receipt_hashes.get(symbol, ())) for symbol in symbols
        },
        "missing_exact_chain_symbols": list(missing_exact_chain_symbols),
        "provider_dte_policy": provider_dte_policy,
        "provider_dte_diagnostics_path": str(provider_dte_diagnostics_path),
        "provider_dte_diagnostics_file_sha256": _sha256_path(provider_dte_diagnostics_path),
        "provider_dte_diagnostic_counts": provider_dte_status_counts,
    }
    if missing_exact_chain_symbols:
        write_group_o_attempt_receipt(
            attempt_receipt_path,
            {
                **attempt_receipt,
                "status": "pending_exact_chain",
                "retry_after_utc": (completed_at + GROUP_O_RETRY_DELAY).isoformat(),
            },
        )
        raise GroupOAcquisitionPending(
            "Group O exact chain is not yet available for every frozen symbol"
        )
    if completed_at >= signal_open:
        write_group_o_attempt_receipt(
            attempt_receipt_path,
            {**attempt_receipt, "status": "blocked_signal_open_cutoff"},
        )
        raise GroupOAcquisitionPending(
            "Group O exact-chain acquisition completed after the signal session open"
        )
    destination = Path(output_path)
    candidate_path = cache / "candidate_package.json"
    candidate_atm_iv_by_symbol: dict[str, float] = {}
    candidate_package = build_group_o_session_package(
        signal_session=signal_session,
        symbols=symbols,
        canonical_options_by_symbol=options_by_symbol,
        daily_bars=pd.concat(daily_frames, ignore_index=True),
        source_receipt_hashes_by_symbol=receipt_hashes,
        feature_manifest_path=feature_manifest_path,
        regime_mapping_path=regime_mapping_path,
        output_path=candidate_path,
        implied_movement_atm_iv_by_symbol_out=candidate_atm_iv_by_symbol,
    )
    package = candidate_package
    published_revision_id: str | None = None
    published_at: datetime
    if supersedes_path is not None:
        if Path(supersedes_path) != destination:
            raise ValueError("Group O superseded path must be the exact base package")
        previous = load_group_o_session_package(
            context_root=destination.parents[1],
            signal_session=signal_session,
        )
        has_missing_exact_chain = any(
            context.quality_status == "missing_exact_chain" for context in previous.contexts
        )
        package = _preserve_resolved_group_o_contexts(
            previous=previous,
            candidate=candidate_package,
            include_implied_movement_corrections=not has_missing_exact_chain,
        )
        write_immutable_bytes(
            cache / "revision_candidate_package.json",
            (package.model_dump_json(indent=2) + "\n").encode("utf-8"),
            conflict_message="immutable Group O revision candidate differs",
        )
        try:
            revision = append_group_o_session_revision(
                context_root=destination.parents[1],
                revised_package=package,
                implied_movement_atm_iv_by_symbol=(
                    {}
                    if has_missing_exact_chain
                    else _implied_movement_atm_iv_sources(
                        previous,
                        candidate_atm_iv_by_symbol,
                    )
                ),
                clock=clock,
            )
            if has_missing_exact_chain and any(
                requires_group_o_implied_movement_correction(context)
                for context in revision.package.contexts
            ):
                package = _preserve_resolved_group_o_contexts(
                    previous=revision.package,
                    candidate=candidate_package,
                    include_implied_movement_corrections=True,
                )
                write_immutable_bytes(
                    cache / "implied_movement_revision_candidate_package.json",
                    (package.model_dump_json(indent=2) + "\n").encode("utf-8"),
                    conflict_message=(
                        "immutable Group O implied-movement revision candidate differs"
                    ),
                )
                revision = append_group_o_session_revision(
                    context_root=destination.parents[1],
                    revised_package=package,
                    implied_movement_atm_iv_by_symbol=(
                        _implied_movement_atm_iv_sources(
                            revision.package,
                            candidate_atm_iv_by_symbol,
                        )
                    ),
                    clock=clock,
                )
        except GroupORevisionCutoffError as exc:
            write_group_o_attempt_receipt(
                attempt_receipt_path,
                {
                    **attempt_receipt,
                    "status": "blocked_signal_open_cutoff",
                    "publication_attempt_at_utc": exc.observed_at_utc.isoformat(),
                },
            )
            raise GroupOAcquisitionPending(
                "Group O exact-chain acquisition completed after the signal session open"
            ) from exc
        published_revision_id = revision.revision_id
        published_at = revision.created_at_utc
    else:
        published_at = clock().astimezone(UTC)
        if published_at >= signal_open:
            write_group_o_attempt_receipt(
                attempt_receipt_path,
                {
                    **attempt_receipt,
                    "status": "blocked_signal_open_cutoff",
                    "publication_attempt_at_utc": published_at.isoformat(),
                },
            )
            raise GroupOAcquisitionPending(
                "Group O exact-chain acquisition completed after the signal session open"
            )

        def require_base_link_preopen() -> None:
            nonlocal published_at
            observed = clock()
            if observed.tzinfo is None or observed.utcoffset() is None:
                raise ValueError("Group O publication timestamp must be timezone-aware")
            published_at = observed.astimezone(UTC)
            if published_at >= signal_open:
                raise GroupORevisionCutoffError(
                    observed_at_utc=published_at,
                    signal_open_utc=signal_open,
                )

        try:
            write_immutable_bytes(
                destination,
                candidate_path.read_bytes(),
                conflict_message="immutable Group O package differs",
                before_link=require_base_link_preopen,
            )
        except GroupORevisionCutoffError as exc:
            write_group_o_attempt_receipt(
                attempt_receipt_path,
                {
                    **attempt_receipt,
                    "status": "blocked_signal_open_cutoff",
                    "publication_attempt_at_utc": exc.observed_at_utc.isoformat(),
                },
            )
            raise GroupOAcquisitionPending(
                "Group O exact-chain acquisition completed after the signal session open"
            ) from exc
    write_group_o_attempt_receipt(
        attempt_receipt_path,
        {
            **attempt_receipt,
            "status": "published_revision" if published_revision_id else "published_base",
            "published_at_utc": published_at.isoformat(),
            "published_base_path": str(destination),
            "published_revision_id": published_revision_id,
        },
    )
    return GroupOAcquisitionResult(
        output_path=destination,
        signal_session=signal_session,
        observation_session=observation_session,
        symbol_count=len(symbols),
        canonical_option_rows=accepted_rows,
        rejected_option_rows=rejected_rows,
    )


class EODHDGroupOPreparationService:
    """Prepare the newest due D-1 package without blocking IBKR acquisition on failure."""

    def __init__(
        self,
        *,
        symbols: tuple[str, ...],
        context_root: str | Path,
        cache_root: str | Path,
        feature_manifest_path: str | Path,
        regime_mapping_path: str | Path,
        capture_delay_seconds: int,
        heartbeat: Callable[[], object] | None = None,
        retry_delay: timedelta = GROUP_O_RETRY_DELAY,
    ) -> None:
        self.symbols = symbols
        self.context_root = Path(context_root)
        self.cache_root = Path(cache_root)
        self.feature_manifest_path = Path(feature_manifest_path)
        self.regime_mapping_path = Path(regime_mapping_path)
        self.capture_delay = timedelta(seconds=capture_delay_seconds)
        self.heartbeat = heartbeat
        if retry_delay != GROUP_O_RETRY_DELAY:
            raise ValueError("Group O retry interval is frozen at 15 minutes")
        self.retry_delay = retry_delay
        self.last_error: str | None = None
        self._retry_after: datetime | None = None
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="stocker-group-o",
        )
        self._future: Future[GroupOAcquisitionResult] | None = None
        self._stop = Event()

    def _due_signal_session(self, now: datetime) -> date | None:
        now_utc = now.astimezone(UTC)
        local_day = now_utc.astimezone(NEW_YORK).date()
        schedule = _xnys_schedule(local_day - timedelta(days=10), local_day + timedelta(days=10))
        sessions = [date.fromisoformat(str(index)[:10]) for index in schedule.index]
        due: list[date] = []
        for observation, signal_session in zip(sessions[:-1], sessions[1:], strict=True):
            close_value = cast(
                Any,
                schedule.loc[observation.isoformat(), "market_close"],
            )
            closed = pd.Timestamp(close_value).to_pydatetime().astimezone(UTC)
            if closed + self.capture_delay <= now_utc:
                due.append(signal_session)
        return None if not due else due[-1]

    @staticmethod
    def _requires_source_revision(package: FrozenGroupOSessionPackage) -> bool:
        return any(
            context.quality_status == "missing_exact_chain"
            or requires_group_o_implied_movement_correction(context)
            for context in package.contexts
        )

    def poll(self, *, now: datetime) -> GroupOAcquisitionResult | None:
        now_utc = now.astimezone(UTC)
        if self._future is not None:
            if not self._future.done():
                return None
            try:
                completed = self._future.result()
            except Exception as exc:
                self.last_error = str(exc)
                self._retry_after = now_utc + self.retry_delay
                completed = None
            else:
                self.last_error = None
                self._retry_after = None
            self._future = None
            if completed is not None:
                return completed
        if self._retry_after is not None and now_utc < self._retry_after:
            return None
        signal_session = self._due_signal_session(now_utc)
        if signal_session is None:
            return None
        output = self.context_root / "group-o" / f"{signal_session.isoformat()}.json"
        supersedes_path: Path | None = None
        if output.is_file():
            try:
                resolved = load_group_o_session_package(
                    context_root=self.context_root,
                    signal_session=signal_session,
                )
            except ValueError as exc:
                self.last_error = f"blocked_invalid_group_o_package: {exc}"
                self._retry_after = None
                return None
            if not self._requires_source_revision(resolved):
                self.last_error = None
                self._retry_after = None
                return None
            supersedes_path = output
        observation_session = previous_xnys_session(signal_session)
        signal_open, _ = xnys_session_bounds(signal_session)
        if now_utc >= signal_open:
            self.last_error = (
                "blocked_group_o_acquisition_cutoff: exact chain remained unavailable "
                "at signal-session open"
            )
            self._retry_after = None
            return None
        persisted_retry = group_o_retry_not_before(
            cache_root=self.cache_root,
            observation_session=observation_session,
        )
        if persisted_retry is not None and now_utc < persisted_retry:
            self.last_error = f"deferred_group_o_retry_not_before: {persisted_retry.isoformat()}"
            self._retry_after = persisted_retry
            return None
        attempt_id, _ = allocate_group_o_attempt(
            cache_root=self.cache_root,
            observation_session=observation_session,
        )
        self._future = self._executor.submit(
            acquire_eodhd_group_o_session_package,
            signal_session=signal_session,
            symbols=self.symbols,
            output_path=output,
            cache_root=self.cache_root,
            cache_attempt_id=attempt_id,
            feature_manifest_path=self.feature_manifest_path,
            regime_mapping_path=self.regime_mapping_path,
            supersedes_path=supersedes_path,
            heartbeat=self.heartbeat,
            cancellation_requested=self._stop.is_set,
        )
        return None

    def shutdown(self) -> None:
        self._stop.set()
        self._executor.shutdown(wait=True, cancel_futures=True)


__all__ = [
    "EODHDGroupOPreparationService",
    "FrozenGroupOArtifacts",
    "GROUP_O_FEATURES",
    "GROUP_O_RETRY_DELAY",
    "GroupOAcquisitionResult",
    "GroupOAcquisitionPending",
    "HistoricalActivityBuildResult",
    "acquire_eodhd_historical_activity_baseline",
    "acquire_eodhd_group_o_session_package",
    "allocate_group_o_attempt",
    "build_group_o_session_package",
    "build_historical_activity_baseline",
    "group_o_retry_not_before",
    "load_group_o_attempt_receipt",
    "signed_group_o_attempt_receipt",
    "write_group_o_attempt_receipt",
]
