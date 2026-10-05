"""Completed Saxo bars, reference sessions and the display-only daily range."""

from __future__ import annotations

import asyncio
import logging
import math
import time
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from stocker_execution.contracts import future_identity, utc
from stocker_execution.reference_sessions import load_selections
from stocker_execution.rules import NY, Bar, reference_summary, session_day
from stocker_execution.saxo_auth import SaxoError
from stocker_execution.saxo_stream import ChartState

if TYPE_CHECKING:
    from stocker_execution.saxo_data import DataService, MarketState

log = logging.getLogger(__name__)
REFERENCE_RETRY_SECONDS = 900
MAX_FILLED_GAP_MINUTES = 5  # longest run of no-trade minutes counted as unchanged


def bar_from_sample(row: dict[str, Any]) -> Bar:
    """One Saxo chart sample as a Bar; KeyError/TypeError/ValueError for an unusable sample.

    Volume absence is a strategy block, not a synthetic zero.
    """
    interest, state = row.get("Interest"), row.get("MarketTradingState")
    return Bar(
        utc(row["Time"]),
        float(row["Open"]),
        float(row["High"]),
        float(row["Low"]),
        float(row["Close"]),
        float(row["Volume"]),
        interest=float(interest) if isinstance(interest, (int, float)) else None,
        state=state if isinstance(state, str) else None,
    )


def bar_fields(bar: Bar) -> dict[str, float]:
    return {k: getattr(bar, k) for k in ("open", "high", "low", "close", "volume")}


def completed_bars(raw: dict[str, Any], at: datetime) -> list[Bar]:
    rows = sorted(raw.get("Data", []), key=lambda b: b.get("Time", ""))
    result = []
    sent = set()  # minutes Saxo did send, valid or not: only omitted minutes may be filled
    # A later sample proves the mutable tail has rolled, even when the wall clock has advanced.
    for row in rows[:-1]:
        try:
            stamp = utc(row["Time"])
            sent.add(stamp)
            if stamp + timedelta(minutes=1) > at:
                continue
            b = bar_from_sample(row)
            if b.valid():
                result.append(b)
        except (KeyError, TypeError, ValueError):
            continue
    return fill_short_gaps(result, sent)


def streamed_bar(chart: ChartState, minute: datetime, at: datetime) -> Bar | None:
    """The completed streamed sample for `minute`, or None.

    Complete means a later sample has started (Saxo opens the next sample when the minute ends)
    and the minute has ended on the wall clock: the rule that drops the REST read's newest
    sample. The stream must be current: a snapshot, no gap or delay, contact inside its timeout.
    """
    if (
        chart.problem
        or chart.last_contact is None
        or not 0 <= at.timestamp() - chart.last_contact <= chart.inactivity_timeout
        or minute + timedelta(minutes=1) > at
    ):
        return None
    try:
        by_time = {utc(t): s for t, s in chart.samples.items()}
        sample = by_time.get(minute)
        if sample is None or not any(t > minute for t in by_time):
            return None
        bar = bar_from_sample(sample)
    except (KeyError, TypeError, ValueError):
        return None
    return bar if bar.valid() else None


def confirm_streamed_bars(data: DataService, state: MarketState, bars: list[Bar]) -> None:
    """Every boundary bar a decision took from the stream is checked against the next REST read
    of that minute. Any difference trips the stream until a restart: decisions use REST alone
    and the clock's evidence records both bars. Pending checks live in memory: a restart in
    the minute between a stream-decided clock and that REST read drops the check, and the
    clock's `boundary_bar` detail is what remains for an offline comparison."""
    if not state.stream_bars:
        return
    by_time = {b.at: b for b in bars}
    newest = max(by_time) if by_time else None
    for minute, (streamed, event_id) in list(state.stream_bars.items()):
        rest = by_time.get(minute)
        if rest is None and (newest is None or newest <= minute):
            continue  # the REST read has not reached that minute yet
        del state.stream_bars[minute]
        evidence = {
            "market": state.market,
            "minute": minute.isoformat(),
            "event_id": event_id,
            "streamed": bar_fields(streamed),
            "rest": bar_fields(rest) if rest else None,
        }
        if rest is not None and bar_fields(rest) == bar_fields(streamed):
            state.chart_confirmed += 1
            continue
        data.chart_stream_problem = "CHART_STREAM_MISMATCH"
        data.chart_mismatch = evidence
        data.chart_audits.append((event_id, evidence))
        log.warning("CHART_STREAM_MISMATCH %s %s", state.market, minute.isoformat())


def fill_short_gaps(bars: list[Bar], sent: set[datetime]) -> list[Bar]:
    """Saxo omits minutes in which nothing traded. The user's rule (2026-10-01): a gap of at most
    MAX_FILLED_GAP_MINUTES between two real bars, with no sample sent for those minutes, is that
    many unchanged minutes (the previous close, zero volume). A longer gap may be an outage, and a
    sent-but-invalid sample is bad data, so both stay missing and still block."""
    out: list[Bar] = []
    for bar in bars:
        if out:
            prior = out[-1]
            missing = int((bar.at - prior.at).total_seconds() // 60) - 1
            minutes = [prior.at + timedelta(minutes=k) for k in range(1, missing + 1)]
            if 1 <= missing <= MAX_FILLED_GAP_MINUTES and not sent.intersection(minutes):
                c = prior.close
                out.extend(Bar(m, c, c, c, c, 0.0) for m in minutes)
        out.append(bar)
    return out


async def history(data: DataService, state: MarketState, *, boundary: bool = False) -> None:
    if not state.identity:
        return
    result = await data.client.request(
        "GET",
        "/chart/v3/charts",
        params={
            "Uic": state.identity["uic"],
            "AssetType": "ContractFutures",
            "Horizon": 1,
            "Count": 1200,
            "FieldGroups": "Data,ChartInfo",
        },
    )
    state.capabilities["history"] = {
        "first_sample": (result.get("ChartInfo") or {}).get("FirstSampleTime"),
        "data_version": result.get("DataVersion"),
        "returned_samples": len(result.get("Data", [])),
        "delayed_by_minutes": (result.get("ChartInfo") or {}).get("DelayedByMinutes"),
        "semantics": "Saxo chart samples; mutable tail excluded; no quote reconstruction",
    }
    bars = completed_bars(result, datetime.now(UTC))
    confirm_streamed_bars(data, state, bars)
    state.bars = bars
    state.history_checked = time.monotonic()
    try:
        await asyncio.to_thread(
            data.bar_cache.save, state.identity, bars, result.get("DataVersion")
        )
    except (OSError, ValueError):
        data.bar_cache.problem = "BAR_STORAGE_UNAVAILABLE"
    state.history_problem = ""
    if not bars:
        state.history_problem = "SAXO_COMPLETED_OHLCV_UNAVAILABLE"
        return
    delay = state.capabilities["history"]["delayed_by_minutes"]
    if delay != 0:
        # Delayed bars can never supply the final completed minute inside the entry
        # deadline; an unreported delay is unknown, and unknown blocks (as for quotes).
        state.history_problem = (
            "SAXO_CHART_DATA_DELAYED"
            if isinstance(delay, (int, float))
            else "SAXO_CHART_DELAY_UNKNOWN"
        )
        return
    if boundary:
        return
    day = session_day(datetime.now(NY))
    if state.reference_day == day.isoformat():
        return
    if time.monotonic() < state.reference_retry_at:
        # Entries stay blocked; do not refetch every reference session each minute.
        state.history_problem = state.reference_failure
        return
    state.references = []
    if not data.config.reference_selections_file:
        state.history_problem = "REFERENCE_SESSION_CONTRACT_SELECTION_UNVERIFIED"
        return
    try:
        audit, digest = load_selections(
            data.config.reference_selections_file,
            data.config.data_environment,
            state.market,
            day,
            state.identity["uic"],
        )
        references = []
        for selected in audit.sessions:
            raw = await data.client.request(
                "GET",
                f"/ref/v1/instruments/details/{selected.contract.uic}/ContractFutures",
                params={"AccountKey": data.client.oauth.account_key},
            )
            identity = future_identity(state.market, data.config.data_environment, raw)
            if (
                identity["symbol"] != selected.contract.symbol
                or identity["exchange"] != selected.contract.exchange
                or identity["contract_month"] != selected.contract.contract_month
            ):
                raise ValueError("REFERENCE_SESSION_IDENTITY_MISMATCH")
            start = datetime.combine(selected.day, datetime.min.time(), NY) + timedelta(hours=8)
            prior = await history_range(
                data, selected.contract.uic, start, start + timedelta(hours=9)
            )
            if len(prior) != 540:
                raise ValueError("REFERENCE_SESSION_OHLCV_COVERAGE_INCOMPLETE")
            # The same session's overnight part (18:00 the evening before to 08:00) feeds only
            # the overnight hours' medians: a gap there leaves those hours without a median (their
            # clocks stay blocked) and never blocks the 08:00-17:00 part above (2026-10-04).
            evening = datetime.combine(
                selected.day - timedelta(days=1), datetime.min.time(), NY
            ) + timedelta(hours=18)
            overnight = await history_range(data, selected.contract.uic, evening, start)
            references.append(reference_summary(overnight + prior))
        state.references, state.reference_day = references, day.isoformat()
        state.capabilities["reference_selection_audit_sha256"] = digest
    except (OSError, ValueError):
        state.history_problem = "REFERENCE_SESSION_AUDIT_OR_SAXO_COVERAGE_UNVERIFIED"
        state.reference_failure = state.history_problem
        state.reference_retry_at = time.monotonic() + REFERENCE_RETRY_SECONDS


async def daily_context(data: DataService, state: MarketState) -> None:
    """Display only: 20-session average daily range from completed Saxo daily samples."""
    day = datetime.now(NY).date().isoformat()
    if not state.identity or state.daily_day == day:
        return
    state.daily_day = day
    try:
        result = await data.client.request(
            "GET",
            "/chart/v3/charts",
            params={
                "Uic": state.identity["uic"],
                "AssetType": "ContractFutures",
                "Horizon": 1440,
                "Count": 21,
                "FieldGroups": "Data",
            },
        )
        rows = sorted(result.get("Data", []), key=lambda r: r.get("Time", ""))[:-1]
        ranges = [float(r["High"]) - float(r["Low"]) for r in rows[-20:]]
        if len(ranges) < 5 or any(not math.isfinite(x) or x < 0 for x in ranges):
            raise ValueError("DAILY_SAMPLES_INSUFFICIENT")
        state.daily_range, state.daily_problem = sum(ranges) / len(ranges), ""
    except (SaxoError, ValueError, KeyError, TypeError) as exc:
        state.daily_range = None
        state.daily_problem = str(exc) if isinstance(exc, ValueError) else "DAILY_SCHEMA"


async def history_range(
    data: DataService,
    uic: int,
    start: datetime,
    end: datetime,
) -> list[Bar]:
    samples: dict[str, dict[str, Any]] = {}
    cursor, version = start, None
    for _ in range(8):  # pages of 1,200 minutes: more than a full session
        result = await data.client.request(
            "GET",
            "/chart/v3/charts",
            params={
                "Uic": uic,
                "AssetType": "ContractFutures",
                "Horizon": 1,
                "Count": 1200,
                "Mode": "From",
                "Time": cursor.isoformat(),
                "FieldGroups": "Data,ChartInfo",
            },
        )
        if version is not None and result.get("DataVersion") != version:
            raise SaxoError("CHART_VERSION_CHANGED_REFETCH_REQUIRED")
        version = result.get("DataVersion")
        rows = result.get("Data", [])
        if not rows:
            break
        samples.update({r["Time"]: r for r in rows})
        newest = max(utc(r["Time"]) for r in rows)
        if newest >= end:
            break
        if newest <= cursor:
            raise SaxoError("CHART_PAGINATION_DID_NOT_ADVANCE")
        cursor = newest
    return [
        b
        for b in completed_bars({"Data": list(samples.values())}, end + timedelta(minutes=1))
        if start <= b.at < end
    ]
