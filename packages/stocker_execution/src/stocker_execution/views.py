"""Display-only summaries for the dashboard. Nothing here feeds admission or management.

Gates and the setup checklist summarise state at the last refresh; the clock decision
re-checks everything itself when it runs.
"""

import json
import math
import time
import zlib
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from stocker_execution.config import MARKETS, MAX_PREMIUM_RISK_GBP, QUOTE_MAX_AGE_SECONDS
from stocker_execution.contracts import quote_check, session_state, utc
from stocker_execution.rules import NY, clocks

if TYPE_CHECKING:
    from stocker_execution.recorder import Recorder
    from stocker_execution.runtime import Runtime
    from stocker_execution.saxo_data import MarketState
    from stocker_execution.saxo_stream import PriceState

# GC is optional: its options are reduce-only on the account (temporary, Saxo 2026-10-04).
TRADEABLE = ("CL", "ES", "NQ")


def quote_current(state: "MarketState", at: float, stream_at: float | None) -> bool:
    receipt = state.price.standing(state.price.receipt, stream_at)
    return (
        receipt is not None
        and 0 <= at - receipt <= QUOTE_MAX_AGE_SECONDS
        and not state.price.problem
    )


def price_problem(price: "PriceState", at: float, stream_at: float | None) -> str:
    """The same check a clock decision applies: current, real-time, usable price."""
    if price.problem:
        return price.problem
    try:
        receipt = price.standing(price.receipt, stream_at)
        quote_check(price.value or {}, receipt, datetime.fromtimestamp(at, UTC))
    except ValueError as exc:
        return str(exc)
    return ""


def gates(
    runtime: "Runtime", market: str, state: "MarketState", at: float, paused: bool
) -> list[dict[str, Any]]:
    """Ordered readiness checks, mirroring the gates a clock decision applies."""
    oauth = runtime.data.client.oauth.status
    candidate = state.candidate_uic if state.candidate_uic in runtime.data.options else None
    costs = runtime.data.option_view(candidate, at)["costs"] if candidate else {}
    execution = "ENTRIES_PAUSED" if paused else runtime.broker.entry_reason()
    stream = runtime.data.stream_at
    rows = [
        (
            "saxo",
            "Saxo",
            runtime.data.connected and oauth == "AUTHENTICATED",
            runtime.data.problem or oauth,
        ),
        ("contract", "Contract", state.identity is not None, state.problem),
        ("quote", "Quote", not (problem := price_problem(state.price, at, stream)), problem),
        ("fx", "GBP/USD rate", not (fx := price_problem(runtime.data.fx, at, stream)), fx),
        (
            "history",
            "History",
            state.identity is not None and not state.history_problem,
            state.history_problem or "AWAITING_CONTRACT",
        ),
        (
            "approval",
            "Option approval",
            market in runtime.config.mappings,
            "LISTED_PRODUCT_AND_DELTA_TOLERANCE_UNAPPROVED",
        ),
        (
            "strike",
            "Strike",
            candidate is not None and not state.candidate_problem,
            state.candidate_problem or "NO_CANDIDATE",
        ),
        (
            "cost",
            f"Cost ≤ £{MAX_PREMIUM_RISK_GBP:,}",
            costs.get("budget_result") == "WITHIN_BUDGET",
            costs.get("reason") or costs.get("budget_result") or "NO_CANDIDATE",
        ),
        ("execution", "Execution", not execution, execution),
    ]
    return [
        {"key": k, "label": label, "ok": bool(ok), "detail": "" if ok else (detail or "UNVERIFIED")}
        for k, label, ok, detail in rows
    ]


def setup(runtime: "Runtime") -> list[dict[str, Any]]:
    """What still has to happen before the strategy can take a paper trade."""
    config, data = runtime.config, runtime.data
    states = list(runtime.markets.values())
    verified = sum(s.identity is not None for s in states)
    references = sum(len(s.references) >= 5 for s in states)
    approved = [m for m in MARKETS if m in config.mappings]
    delays = [
        ((s.price.value or {}).get("Quote") or {}).get("DelayedByMinutes")
        for s in states
        if s.price.value
    ]
    known = [d for d in delays if isinstance(d, (int, float))]
    realtime = bool(delays) and len(known) == len(delays) and all(d == 0 for d in known)
    delayed = max(known, default=None)
    realtime_detail = (
        f"Saxo reports quotes delayed by {delayed:g} min; the strategy needs real-time "
        "exchange data (entitlement or LIVE data with INTERNAL_PAPER)"
        if delayed
        else "Saxo did not report the quote delay; unknown delay blocks entries"
        if delays
        else "No futures quote received yet"
    )
    oauth = data.client.oauth.status
    items = [
        ("oauth", "Saxo login", oauth == "AUTHENTICATED", oauth, False),
        (
            "account",
            "Account verified",
            data.account_verified,
            "Connect Saxo and verify the configured account",
            False,
        ),
        ("stream", "Price stream", data.connected, data.problem, False),
        (
            "fx",
            "GBP/USD conversion rate",
            data.fx.receipt is not None and not data.fx.problem,
            data.fx.problem or "Awaiting the GBPUSD quote",
            False,
        ),
        ("realtime", "Real-time market data", realtime, realtime_detail, False),
        (
            "contracts",
            f"Futures contracts verified {verified}/{len(MARKETS)}",
            verified == len(MARKETS),
            f"{len(config.contracts)}/{len(MARKETS)} configured; "
            "select from Saxo ContractFutures data",
            False,
        ),
        (
            "references",
            f"Reference sessions verified {references}/{len(MARKETS)}",
            references == len(MARKETS),
            "Set reference_selections_file"
            if not config.reference_selections_file
            else "Awaiting five verified prior sessions per market",
            False,
        ),
        (
            "approvals",
            f"Option approvals {len(approved)}/{len(MARKETS)}",
            all(m in config.mappings for m in TRADEABLE),
            "Approve product, delta tolerance, fees and expiry times (GC optional)",
            False,
        ),
        (
            "mode",
            "Paper execution mode",
            config.execution_mode != "DISABLED",
            "execution_mode is DISABLED",
            False,
        ),
        (
            "armed",
            "Armed after preflight",
            runtime.broker.armed,
            "Run preflight, then arm explicitly",
            False,
        ),
        (
            "recording",
            "Event recording",
            config.recorder.persistent_capture,
            "Optional: needs recording-permission evidence",
            True,
        ),
        ("alerts", "Alerts", runtime.alerts.enabled, "Optional: configure an alert URL file", True),
    ]
    return [
        {
            "key": k,
            "label": label,
            "done": bool(done),
            "detail": "" if done else detail,
            "optional": optional,
        }
        for k, label, done, detail, optional in items
    ]


def chart_context(state: "MarketState", at: datetime) -> dict[str, Any]:
    """Clock marks for today, the RV15 window and session state at each clock."""
    today = clocks(at.astimezone(NY).date())
    context: dict[str, Any] = {
        "clocks": [
            {
                "at": clock.isoformat(),
                "session": session_state(state.reference, clock) if state.reference else None,
            }
            for clock in today
        ],
        "rv_window": None,
    }
    if state.bars:
        end = state.bars[-1].at + timedelta(minutes=1)
        context["rv_window"] = [(end - timedelta(minutes=15)).isoformat(), end.isoformat()]
    return context


def sessions_today(state: "MarketState", at: datetime) -> list[dict[str, Any]]:
    """Saxo trading sessions overlapping today's New York date, as reported."""
    start = datetime.combine(at.astimezone(NY).date(), datetime.min.time(), NY).astimezone(UTC)
    end = start + timedelta(days=1)
    result = []
    for session in (state.reference.get("TradingSessions") or {}).get("Sessions", []):
        try:
            begin, finish = utc(session["StartTime"]), utc(session["EndTime"])
        except (KeyError, ValueError, TypeError):
            continue
        if begin < end and finish > start:
            result.append(
                {
                    "start": begin.isoformat(),
                    "end": finish.isoformat(),
                    "state": str(session.get("State", "UNKNOWN")),
                }
            )
    return result[:12]


def timeline(
    signal: dict[str, Any], lifecycle: list[dict[str, Any]], fills: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """One opportunity as ordered steps: clock, checks, option, admission, orders, outcome."""
    detail = json.loads(signal["detail"])
    inputs = detail.get("inputs") or {}
    option = detail.get("option_context") or {}
    identity = option.get("identity") or {}
    steps: list[dict[str, Any]] = [
        {
            "at": signal["signal_at"],
            "kind": "CLOCK",
            "ok": True,
            "title": f"Frozen clock · {signal['market']}",
            "detail": f"rule {signal['rule_version']} · exit anchor {signal['exit_at']}",
        },
        {
            "at": signal["signal_at"],
            "kind": "CHECKS",
            "ok": not detail.get("skip_reason"),
            "title": "Data and eligibility checks",
            "detail": detail.get("skip_reason")
            or " · ".join(f"{k} {v:.6g}" for k, v in inputs.items() if isinstance(v, (int, float))),
        },
        {
            "at": signal["signal_at"],
            "kind": "OPTION",
            "ok": option.get("selection_status") == "SELECTED" if option else None,
            "title": "Option selection",
            "detail": (
                f"{identity.get('right', '')} {identity.get('strike', '')} · "
                f"UIC {identity.get('uic')}"
            )
            if identity
            else option.get("reason", "Not reached"),
        },
    ]
    titles = {
        "ADMISSION": "Admission",
        "DURABLE_SUBMISSION_INTENT": "Order intent recorded",
        "INTERNAL_SIMULATED_FILL": "Internal paper fill",
        "SAXO_SIM_BROKER_FILL": "Saxo SIM fill",
        "SAXO_SIM_ORDER_EVIDENCE": "Broker order evidence",
        "DECISION": "Decision",
        "CLOSURE_CONFIRMED": "Closure confirmed",
    }
    for row in sorted(lifecycle, key=lambda r: r["sequence"]):
        body = json.loads(row["detail"])
        if row["kind"] == "ADMISSION":
            ok, text = (
                not body.get("reason"),
                body.get("reason")
                or (f"reserved · cost £{(body.get('plan') or {}).get('total_gbp', '?')}"),
            )
        elif row["kind"] == "DURABLE_SUBMISSION_INTENT":
            ok, text = True, f"{body.get('role')} limit {body.get('limit')}"
        elif row["kind"] in {"INTERNAL_SIMULATED_FILL", "SAXO_SIM_BROKER_FILL"}:
            ok, text = True, f"{body.get('side')} {body.get('quantity')} @ {body.get('price')}"
        elif row["kind"] == "DECISION":
            ok = body.get("decision") not in {
                "SKIPPED",
                "EXPOSURE_EXCEPTION",
                "ORDER_STATUS_UNCERTAIN",
            }
            text = " · ".join(x for x in (body.get("decision"), body.get("reason")) if x)
        elif row["kind"] == "SAXO_SIM_ORDER_EVIDENCE":
            ok, text = True, f"{body.get('Status')} filled {body.get('FilledAmount')}"
        else:
            ok, text = True, ""
        steps.append(
            {
                "at": row["at"],
                "kind": row["kind"],
                "ok": ok,
                "title": titles.get(row["kind"], row["kind"]),
                "detail": text,
            }
        )
    steps.append(
        {
            "at": None,
            "kind": "OUTCOME",
            "ok": None,
            "title": "Current state",
            "detail": " · ".join(x for x in (signal["decision"], signal["reason"]) if x)
            + (f" · {len(fills)} fill(s)" if fills else ""),
        }
    )
    return steps


_book_cache: dict[str, tuple[float, dict[str, Any]]] = {}


def book_rows(recorder: "Recorder", instrument: str) -> tuple[float | None, list[bytes]]:
    """Snapshot the rolling window on the event loop; decoding happens off-loop."""
    window = recorder.windows.get(instrument)
    if window is None:
        return None, []
    return window.identity.get("tick_size"), [b for _, b, _ in window.rows]


def book_series(
    tick: float | None, blobs: list[bytes], instrument: str, bucket: int = 5
) -> dict[str, Any]:
    """Downsampled sampled-book history from the recorder's own rolling rows.

    Each bucket keeps the last observation inside it. Levels are tick indices and sizes
    exactly as recorded by book_flow; no cancellations, trades or gaps are inferred.
    A five-second series is served for five seconds: the window changes with every message.
    """
    cached = _book_cache.get(instrument)
    if cached and time.monotonic() - cached[0] < 5:
        return cached[1]
    buckets: dict[int, dict[str, Any]] = {}
    for blob in blobs:
        record = json.loads(zlib.decompress(blob))
        flow = record.get("book_flow")
        if not flow:
            continue
        slot = int(flow["at"] // bucket * bucket)
        basis = flow.get("basis") or {}
        depth = flow.get("depth") or {}
        buckets[slot] = {
            "at": slot,
            "status": flow.get("status"),
            "spread_ticks": flow.get("spread_ticks"),
            "imbalance1": (depth.get("1") or {}).get("imbalance"),
            "imbalance5": (depth.get("5") or {}).get("imbalance"),
            "bid": basis.get("bid", [])[:10],
            "ask": basis.get("ask", [])[:10],
        }
    result = {
        "tick_size": tick,
        "bucket_seconds": bucket,
        "series": [buckets[k] for k in sorted(buckets)],
        "semantics": "Sampled Saxo depth, last observation per bucket; not an execution tape",
    }
    _book_cache[instrument] = (time.monotonic(), result)
    return result


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value:
        return None
    return float(value)


def price_context(state: "MarketState") -> dict[str, Any]:
    """Provider-reported session context for the future. Display only."""
    value = state.price.value or {}
    info = value.get("PriceInfo") or {}
    details = value.get("PriceInfoDetails") or {}
    instrument = value.get("InstrumentPriceDetails") or {}
    quote = value.get("Quote") or {}
    return {
        "open": _number(details.get("Open")),
        "high": _number(info.get("High")),
        "low": _number(info.get("Low")),
        "last_close": _number(details.get("LastClose")),
        "net_change": _number(info.get("NetChange")),
        "percent_change": _number(info.get("PercentChange")),
        "open_interest": _number(instrument.get("OpenInterest")),
        "market_state": quote.get("MarketState")
        if isinstance(quote.get("MarketState"), str)
        else None,
        "daily_range": state.daily_range,
        "daily_problem": state.daily_problem,
        "daily_open_interest": state.daily_open_interest,
        "basis": "Saxo-reported session fields; NetChange is mid minus last close",
    }


def smile(
    board: dict[str, Any], day: str, scale: str = "UNVERIFIED", sigma: float | None = None
) -> dict[str, Any] | None:
    """Options-chain snapshot for one expiry: provider IV, delta, OI and indications.

    Raw provider values are always kept. Only with an operator-verified scale is the
    volatility normalised to an annual fraction and compared with the frozen model's
    sigma. Chain prices are indications, never executable quotes.
    """
    divisor = {"FRACTION": 1.0, "PERCENT": 100.0}.get(scale)
    expiries = [e for e in board.get("Expiries") or [] if isinstance(e, dict)]
    expiry = next(
        (e for e in expiries if str(e.get("Expiry", ""))[:10] == day),
        expiries[0] if expiries else None,
    )
    if expiry is None:
        return None

    def side(value: Any) -> dict[str, Any] | None:
        if not isinstance(value, dict):
            return None
        raw_greeks = value.get("Greeks")
        greeks: dict[str, Any] = raw_greeks if isinstance(raw_greeks, dict) else {}
        raw = _number(greeks.get("MidVolatility"))
        iv = raw / divisor if raw is not None and divisor else None
        return {
            "uic": value.get("Uic") if isinstance(value.get("Uic"), int) else None,
            "bid": _number(value.get("Bid")),
            "ask": _number(value.get("Ask")),
            "delta": _number(greeks.get("Delta")),
            "mid_volatility": raw,
            "iv": iv,
            "iv_minus_model": iv - sigma if iv is not None and sigma else None,
            "open_interest": _number(value.get("OpenInterest")),
            "volume": _number(value.get("Volume")),
        }

    rows = [s for s in expiry.get("Strikes") or [] if isinstance(s, dict)]
    # The board keeps every strike Saxo has sent, keyed by index: the first snapshot's lowest
    # strikes stay ahead of the focused window, so show the strikes nearest the money (2026-10-01).
    mid = _number(expiry.get("MidStrikePrice"))
    if mid is not None:

        def distance(row: dict[str, Any]) -> float:
            strike = _number(row.get("Strike"))
            return abs(strike - mid) if strike is not None else math.inf

        rows.sort(key=distance)
    strikes = [
        {
            "strike": _number(s.get("Strike")),
            "mid_volatility_pct": _number(s.get("MidVolatilityPct")),
            "call": side(s.get("Call")),
            "put": side(s.get("Put")),
        }
        for s in rows[:25]
    ]
    return {
        "expiry": expiry.get("Expiry"),
        "last_trade": expiry.get("LastTradeDate"),
        "mid_strike_price": _number(expiry.get("MidStrikePrice")),
        "strikes": sorted(
            (s for s in strikes if s["strike"] is not None), key=lambda s: s["strike"]
        ),
        "scaling": "PROVIDER_NATIVE_UNVERIFIED" if divisor is None else scale,
        "model_sigma": sigma,
        "executable": False,
    }


def model_sigma(rv15: float | None) -> float | None:
    """The frozen model's annualised volatility from RV15 (the frozen strike model)."""
    return rv15 * math.sqrt(525600 / 15) if rv15 else None


def iv_spread(option: dict[str, Any], rv15: float | None) -> dict[str, Any]:
    """Implied (verified price Greeks.MidVol) minus the frozen model's sigma. Context only."""
    raw = ((option.get("analytics") or {}).get("Greeks.MidVol") or {}).get("value")
    sigma = model_sigma(rv15)
    # Saxo's price-subscription Greeks.MidVol is an annual fraction: a live SIM probe on
    # 2026-09-29 read 0.589 (CL) and 0.523 (NG) at the money against chain ImpliedVolatility
    # of 50.5 and 55.3 (percent). The chain's per-strike MidVolatility is also a fraction; see
    # FuturesConfig.provider_volatility_scale.
    iv = _number(raw)
    return {
        "implied_volatility": iv,
        "model_sigma": sigma,
        "iv_minus_model_sigma": iv - sigma if iv is not None and sigma else None,
    }
