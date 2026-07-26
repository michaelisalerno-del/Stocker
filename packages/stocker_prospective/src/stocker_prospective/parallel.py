"""Bounded after-session EODHD capture for prospective source-parity evidence."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from datetime import UTC, date, datetime, timedelta
from typing import Literal, Protocol
from zoneinfo import ZoneInfo

import pandas as pd
from pydantic import BaseModel, ConfigDict

from stocker_data.vendors.eodhd import EODHDClient, normalize_intraday_response
from stocker_prospective.bundle import ANCHOR_COHORT
from stocker_prospective.config import ProspectiveConfig
from stocker_prospective.database import (
    EvidenceMetadata,
    ProspectiveRepository,
    RecorderLeaseHeld,
    SourceBarObservationInput,
)
from stocker_prospective.recorder import RecorderDeploymentIdentity

NEW_YORK = ZoneInfo("America/New_York")
EXPECTED_REGULAR_SESSION_BAR_COUNT = 78


class ParallelCaptureError(RuntimeError):
    """One bounded parity-evidence capture could not be completed."""


class ParallelSourceBar(BaseModel):
    """Normalized EODHD bar that remains permanently ineligible for scoring."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider_record_id: str
    symbol: str
    session_date: date
    bar_start_utc: datetime
    bar_end_utc: datetime
    open: float | None
    high: float | None
    low: float | None
    close: float | None
    activity_value: float | None
    source_timestamp_utc: datetime
    receive_timestamp_utc: datetime
    completeness: Literal["complete", "partial"]


class EODHDClientBoundary(Protocol):
    def require_token(self) -> str: ...

    def fetch_intraday_chunk(
        self,
        *,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> object: ...


class ParallelBarProvider(Protocol):
    def require_credentials(self) -> None: ...

    def fetch_session(
        self,
        *,
        symbol: str,
        session_date: date,
        received_at_utc: datetime,
    ) -> tuple[ParallelSourceBar, ...]: ...


def _session_window(session_date: date) -> tuple[datetime, datetime] | None:
    import pandas_market_calendars as mcal

    schedule = mcal.get_calendar("XNYS").schedule(
        start_date=session_date,
        end_date=session_date,
    )
    if schedule.empty:
        return None
    row = schedule.iloc[0]
    opened = pd.Timestamp(row["market_open"]).to_pydatetime().astimezone(UTC)
    closed = pd.Timestamp(row["market_close"]).to_pydatetime().astimezone(UTC)
    return opened, closed


def _finite(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(str(value))
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


class EODHDParallelBarProvider:
    """Fetch one exact completed XNYS session through the trusted EODHD adapter."""

    def __init__(self, *, client: EODHDClientBoundary) -> None:
        self.client = client

    def require_credentials(self) -> None:
        self.client.require_token()

    def fetch_session(
        self,
        *,
        symbol: str,
        session_date: date,
        received_at_utc: datetime,
    ) -> tuple[ParallelSourceBar, ...]:
        window = _session_window(session_date)
        if window is None:
            raise ParallelCaptureError("requested parallel capture date is not an XNYS session")
        opened, closed = window
        payload = self.client.fetch_intraday_chunk(
            symbol=f"{symbol}.US",
            interval="5m",
            start=opened,
            end=closed,
        )
        try:
            frame = normalize_intraday_response(
                payload,
                symbol=symbol,
                instrument_type="equity",
                interval="5m",
            )
        except Exception as exc:
            raise ParallelCaptureError(
                f"blocked_parallel_source_capture: {symbol} EODHD payload is invalid"
            ) from exc
        frame = frame.loc[
            frame["timestamp"].ge(opened) & frame["timestamp"].lt(closed)
        ].copy()
        output: list[ParallelSourceBar] = []
        for row in frame.itertuples(index=False):
            if not isinstance(row.timestamp, datetime):
                raise ParallelCaptureError(
                    f"blocked_parallel_source_capture: {symbol} timestamp type differs"
                )
            bar_start = row.timestamp.astimezone(UTC)
            offset = (bar_start - opened).total_seconds()
            if offset < 0 or offset % 300 != 0:
                raise ParallelCaptureError(
                    f"blocked_parallel_source_capture: {symbol} bar boundary differs"
                )
            open_value = _finite(row.open)
            high_value = _finite(row.high)
            low_value = _finite(row.low)
            close_value = _finite(row.close)
            values = {
                "open": open_value,
                "high": high_value,
                "low": low_value,
                "close": close_value,
            }
            activity = _finite(row.volume)
            valid_market = (
                open_value is not None
                and high_value is not None
                and low_value is not None
                and close_value is not None
                and high_value >= max(open_value, close_value)
                and low_value <= min(open_value, close_value)
                and high_value >= low_value
            )
            complete = valid_market and activity is not None
            output.append(
                ParallelSourceBar(
                    provider_record_id=(
                        f"{symbol}.US:{bar_start.isoformat().replace('+00:00', 'Z')}"
                    ),
                    symbol=symbol,
                    session_date=session_date,
                    bar_start_utc=bar_start,
                    bar_end_utc=bar_start + timedelta(minutes=5),
                    **values,
                    activity_value=activity,
                    source_timestamp_utc=bar_start,
                    receive_timestamp_utc=received_at_utc.astimezone(UTC),
                    completeness="complete" if complete else "partial",
                )
            )
        return tuple(output)


class ParallelSourceCaptureService:
    """Capture the latest due session once without entering the scoring path."""

    def __init__(
        self,
        *,
        config: ProspectiveConfig,
        repository: ProspectiveRepository,
        identity: RecorderDeploymentIdentity,
        provider: ParallelBarProvider,
        sleep: Callable[[float], None] = time.sleep,
        heartbeat: Callable[[], object] | None = None,
    ) -> None:
        if len(identity.symbols) != 20 or len(set(identity.symbols)) != 20:
            raise ValueError("blocked_frozen_universe_mismatch")
        self.config = config
        self.repository = repository
        self.identity = identity
        self.provider = provider
        self._sleep = sleep
        self._heartbeat = heartbeat
        self._last_credential_failure_date: date | None = None

    def _metadata(
        self,
        now: datetime,
        *,
        source_timestamps: tuple[datetime, ...],
    ) -> EvidenceMetadata:
        return EvidenceMetadata(
            run_id=self.config.runtime.run_id or "",
            prospective_start_utc=self.config.runtime.prospective_start_utc,
            app_version=self.config.runtime.app_version,
            git_commit=self.config.runtime.git_commit,
            model_artifact_id=self.identity.model_artifact_id,
            universe_id=self.identity.universe_id,
            cohort=ANCHOR_COHORT,
            source_timestamps=[
                value.astimezone(UTC).isoformat() for value in source_timestamps
            ],
            recorded_at_utc=max(
                now.astimezone(UTC),
                self.config.runtime.prospective_start_utc.astimezone(UTC),
            ),
        )

    def _latest_due_session(self, now: datetime) -> tuple[date, datetime] | None:
        now_utc = now.astimezone(UTC)
        start_day = self.config.runtime.prospective_start_utc.astimezone(NEW_YORK).date()
        end_day = now_utc.astimezone(NEW_YORK).date()
        import pandas_market_calendars as mcal

        schedule = mcal.get_calendar("XNYS").schedule(
            start_date=start_day,
            end_date=end_day,
        )
        due: list[tuple[date, datetime]] = []
        for index, row in schedule.iterrows():
            opened = pd.Timestamp(row["market_open"]).to_pydatetime().astimezone(UTC)
            closed = pd.Timestamp(row["market_close"]).to_pydatetime().astimezone(UTC)
            if opened < self.config.runtime.prospective_start_utc.astimezone(UTC):
                continue
            capture_at = closed + timedelta(
                seconds=self.config.parallel_validation.capture_delay_seconds
            )
            if capture_at <= now_utc:
                due.append((date.fromisoformat(str(index)[:10]), capture_at))
        return None if not due else due[-1]

    def _fetch_with_lease_heartbeat(
        self,
        *,
        symbol: str,
        session_date: date,
        received_at_utc: datetime,
    ) -> tuple[ParallelSourceBar, ...]:
        """Keep lease ownership current while a synchronous vendor request is pending."""

        executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="stocker-parallel-source",
        )
        future = executor.submit(
            self.provider.fetch_session,
            symbol=symbol,
            session_date=session_date,
            received_at_utc=received_at_utc,
        )
        try:
            while True:
                try:
                    return future.result(
                        timeout=float(self.config.runtime.heartbeat_seconds)
                    )
                except FutureTimeoutError:
                    if self._heartbeat is not None:
                        self._heartbeat()
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

    def poll(self, *, now: datetime) -> None:
        if not self.config.parallel_validation.enabled:
            return
        due = self._latest_due_session(now)
        if due is None:
            return
        session_date, capture_at = due
        run_id = self.config.runtime.run_id or ""
        if self.repository.source_capture_completed(
            run_id=run_id,
            provider="eodhd",
            session_date=session_date,
        ):
            return
        metadata = self._metadata(now, source_timestamps=(capture_at,))
        self.repository.create_run(metadata, mode=self.config.runtime.mode)
        try:
            self.provider.require_credentials()
        except Exception:
            if self._last_credential_failure_date != session_date:
                self.repository.record_data_health_event(
                    metadata,
                    severity="blocker",
                    blocker_code="blocked_missing_eodhd_server_token",
                    component="parallel_feature_validation",
                    message="EODHD parallel capture credential is unavailable",
                    details={
                        "provider": "eodhd",
                        "session_date": session_date.isoformat(),
                        "credential_value_exposed": False,
                        "scoring_allowed": False,
                    },
                )
                self._last_credential_failure_date = session_date
            return

        missing: list[str] = []
        captured_symbols = 0
        bar_count = 0
        session_window = _session_window(session_date)
        if session_window is None:
            raise ParallelCaptureError("requested parallel capture date is not an XNYS session")
        session_open, session_close = session_window
        expected_starts = {
            session_open + timedelta(minutes=5 * offset)
            for offset in range(EXPECTED_REGULAR_SESSION_BAR_COUNT)
        }
        for index, symbol in enumerate(self.identity.symbols):
            if self._heartbeat is not None:
                self._heartbeat()
            try:
                bars = self._fetch_with_lease_heartbeat(
                    symbol=symbol,
                    session_date=session_date,
                    received_at_utc=now,
                )
            except RecorderLeaseHeld:
                raise
            except Exception as exc:
                bars = ()
                missing.append(symbol)
                self.repository.record_data_health_event(
                    metadata,
                    severity="warning",
                    blocker_code="blocked_parallel_source_capture",
                    component="parallel_feature_validation",
                    message=f"{symbol}: {type(exc).__name__}",
                    details={
                        "symbol": symbol,
                        "session_date": session_date.isoformat(),
                        "scoring_allowed": False,
                    },
                )
            complete_symbol = (
                session_close - session_open == timedelta(minutes=390)
                and len(bars) == EXPECTED_REGULAR_SESSION_BAR_COUNT
                and {bar.bar_start_utc for bar in bars} == expected_starts
                and all(
                    bar.symbol == symbol
                    and bar.session_date == session_date
                    and bar.bar_end_utc == bar.bar_start_utc + timedelta(minutes=5)
                    and bar.completeness == "complete"
                    for bar in bars
                )
            )
            if complete_symbol:
                captured_symbols += 1
            elif symbol not in missing:
                missing.append(symbol)
                self.repository.record_data_health_event(
                    metadata,
                    severity="warning",
                    blocker_code="blocked_parallel_source_capture",
                    component="parallel_feature_validation",
                    message=f"{symbol}: incomplete exact 78-bar session",
                    details={
                        "symbol": symbol,
                        "session_date": session_date.isoformat(),
                        "observed_bar_count": len(bars),
                        "expected_bar_count": EXPECTED_REGULAR_SESSION_BAR_COUNT,
                        "scoring_allowed": False,
                    },
                )
            for bar in bars:
                self.repository.record_source_bar_observation(
                    SourceBarObservationInput(
                        metadata=self._metadata(
                            now,
                            source_timestamps=(
                                bar.source_timestamp_utc,
                                bar.receive_timestamp_utc,
                            ),
                        ),
                        provider="eodhd",
                        provider_record_id=bar.provider_record_id,
                        symbol=bar.symbol,
                        session_date=bar.session_date,
                        bar_start_utc=bar.bar_start_utc,
                        bar_end_utc=bar.bar_end_utc,
                        open=bar.open,
                        high=bar.high,
                        low=bar.low,
                        close=bar.close,
                        activity_value=bar.activity_value,
                        activity_semantic_label="eodhd_historical_activity_proxy",
                        source_timestamp_utc=bar.source_timestamp_utc,
                        receive_timestamp_utc=bar.receive_timestamp_utc,
                        completeness=bar.completeness,
                        eligibility=False,
                        rejection_reason="parallel_validation_only",
                    )
                )
                bar_count += 1
            if index + 1 < len(self.identity.symbols):
                self._sleep(60.0 / self.config.parallel_validation.requests_per_minute)
        self.repository.record_source_capture_completion(
            metadata,
            provider="eodhd",
            session_date=session_date,
            status="complete" if not missing else "partial",
            requested_symbol_count=len(self.identity.symbols),
            captured_symbol_count=captured_symbols,
            bar_count=bar_count,
            missing_symbols=tuple(missing),
        )
        self.repository.record_audit_event(
            metadata,
            event_type="parallel_source_capture_completed",
            actor=self.config.runtime.instance_id,
            message="after-session EODHD bars retained only for source-parity evidence",
            payload={
                "provider": "eodhd",
                "session_date": session_date.isoformat(),
                "status": "complete" if not missing else "partial",
                "captured_symbol_count": captured_symbols,
                "bar_count": bar_count,
                "missing_symbols": missing,
                "scoring_allowed": False,
                "outcomes_read": False,
            },
        )


def build_parallel_eodhd_service(
    *,
    config: ProspectiveConfig,
    repository: ProspectiveRepository,
    identity: RecorderDeploymentIdentity,
    heartbeat: Callable[[], object] | None = None,
) -> ParallelSourceCaptureService:
    """Construct the trusted adapter without reading or exposing its environment token."""

    from stocker_core.config import EODHDConfig

    client = EODHDClient(
        config=EODHDConfig(
            enabled=True,
            base_url=config.parallel_validation.base_url,
            api_token_env=config.parallel_validation.api_token_env,
            request_timeout_seconds=(
                config.parallel_validation.request_timeout_seconds
            ),
            max_retries=config.parallel_validation.max_retries,
            save_raw_by_default=False,
        )
    )
    return ParallelSourceCaptureService(
        config=config,
        repository=repository,
        identity=identity,
        provider=EODHDParallelBarProvider(client=client),
        heartbeat=heartbeat,
    )


__all__ = [
    "EODHDParallelBarProvider",
    "ParallelCaptureError",
    "ParallelSourceBar",
    "ParallelSourceCaptureService",
    "build_parallel_eodhd_service",
]
