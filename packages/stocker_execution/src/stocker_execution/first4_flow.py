"""FIRST4 observation V1: immutable evidence and deterministic receipt-time reduction.

No broker calls, order decisions, candle/tick-rule fallbacks or tuple deduplication.
Prices compare with 1e-9 currency units absolute numerical tolerance, never spread
fractions. Broker epoch timestamps have ONE SECOND precision, not receipt precision.
"""

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from math import isclose, isfinite
from typing import Any

VERSION = "FIRST4_QUOTE_MATCH_V1"
PRICE_TOLERANCE = 1e-9
VOLUMES = ("buy_est_volume", "sell_est_volume", "unknown_volume", "excluded_volume")


@dataclass(frozen=True, slots=True)
class FlowEvent:
    capture_id: str
    session: str
    con_id: int
    request_id: int
    generation: int
    sequence: int
    received_at: str
    monotonic_ns: int
    kind: str
    feed_mode: str
    broker_time: int | None = None
    broker_precision: str | None = None
    price: float | None = None
    size: float | None = None
    bid: float | None = None
    ask: float | None = None
    bid_size: float | None = None
    ask_size: float | None = None
    tick_type: int | None = None
    exchange: str = ""
    conditions: str = ""
    attributes: tuple[str, ...] = ()
    source: str = ""
    field_name: str = ""
    reason: str = ""

    def record(self) -> dict[str, Any]:
        return asdict(self)


def metrics(values: dict[str, Any]) -> dict[str, Any]:
    buy, sell, unknown = (values.get(k, 0.0) for k in VOLUMES[:3])
    eligible, classified = buy + sell + unknown, buy + sell
    return {
        **values,
        "eligible_observed_volume": eligible,
        "volume_delta": buy - sell,
        "classified_volume_fraction": classified / eligible if eligible else None,
        "unknown_volume_fraction": unknown / eligible if eligible else None,
        "buy_share_of_classified": buy / classified if classified else None,
        "delta_fraction_of_classified": (buy - sell) / classified if classified else None,
    }


def empty_totals() -> dict[str, Any]:
    return {**dict.fromkeys(VOLUMES, 0.0), "trade_count": 0, "classified_trade_count": 0}


@dataclass
class FlowReducer:
    quote_age_ms: int = 1000
    quote: FlowEvent | None = None
    fields: dict[str, tuple[float | None, int, str]] = field(default_factory=dict)
    bars: dict[str, dict[str, Any]] = field(default_factory=dict)
    totals: dict[str, Any] = field(default_factory=empty_totals)
    last_sequence: int = 0
    last_trade_at: str | None = None
    last_quote_at: str | None = None
    last_received_at: str | None = None
    first_received_at: str | None = None
    market_data_type: int = 0
    gaps: list[dict[str, Any]] = field(default_factory=list)
    reasons: dict[str, float] = field(default_factory=dict)

    def classify(self, event: FlowEvent) -> tuple[str, str, bool]:
        if event.tick_type != 1:
            return "EXCLUDED", "NON_LAST_STREAM", False
        if event.attributes or event.conditions:
            return "EXCLUDED", "ATTRIBUTES_OR_UNSUPPORTED_CONDITIONS", False
        if event.price is None or not isfinite(event.price) or event.price <= 0:
            return "EXCLUDED", "INVALID_PRICE", False
        if (
            event.broker_time is not None
            and datetime.fromisoformat(event.received_at).timestamp() - event.broker_time
            > self.quote_age_ms / 1000 + 1
        ):
            return "EXCLUDED", "DEMONSTRABLY_LATE_PRINT", False
        quote = self.quote
        if quote is None:
            return "UNKNOWN", "MISSING_QUOTE", False
        if quote.sequence >= event.sequence or quote.generation != event.generation:
            return "UNKNOWN", "ORDER_OR_GENERATION", False
        coarse = quote.broker_time is not None and quote.broker_time == event.broker_time
        if (
            quote.broker_time is not None
            and event.broker_time is not None
            and quote.broker_time > event.broker_time
        ):
            return "UNKNOWN", "FUTURE_QUOTE", coarse
        if quote.attributes:
            return "UNKNOWN", "QUOTE_ATTRIBUTES", coarse
        if event.feed_mode == "TBT_TRADES_L1_QUOTES" and self.market_data_type != 1:
            return "UNKNOWN", "L1_NOT_CONFIRMED_LIVE", coarse
        sides = [self.fields.get(k) for k in ("bid", "ask")]
        if any(
            x is None or not 0 <= (event.monotonic_ns - x[1]) / 1e6 <= self.quote_age_ms
            for x in sides
        ):
            return "UNKNOWN", "STALE_OR_MISSING_SIDES", coarse
        bid, ask = (x[0] if x else None for x in sides)
        if (
            bid is None
            or ask is None
            or not all(isfinite(v) and v > 0 for v in (bid, ask))
            or bid >= ask
        ):
            return "UNKNOWN", "INVALID_LOCKED_OR_CROSSED", coarse
        at_bid = isclose(event.price, bid, rel_tol=0, abs_tol=PRICE_TOLERANCE)
        at_ask = isclose(event.price, ask, rel_tol=0, abs_tol=PRICE_TOLERANCE)
        if at_bid == at_ask:
            return "UNKNOWN", "AMBIGUOUS_OR_NOT_AT_QUOTE", coarse
        return ("BUY_EST" if at_ask else "SELL_EST"), "QUOTE_MATCH", coarse

    def minute_bar(self, event: FlowEvent) -> dict[str, Any]:
        minute = (
            datetime.fromisoformat(event.received_at)
            .astimezone(UTC)
            .replace(second=0, microsecond=0)
            .isoformat()
        )
        if minute not in self.bars:
            if len(self.bars) >= 400:
                raise ValueError("REGULAR_SESSION_BAR_LIMIT")
            self.bars[minute] = {
                **empty_totals(),
                "minute": minute,
                "price": None,
                "coarse_timestamp_trades": 0,
                "has_gap": bool(self.gaps and self.gaps[-1].get("end_at") is None),
                "first_event_at": event.received_at,
                "last_event_at": event.received_at,
            }
        self.bars[minute]["last_event_at"] = event.received_at
        if self.gaps and self.gaps[-1].get("end_at") is None:
            self.bars[minute]["has_gap"] = True
        return self.bars[minute]

    def apply(self, event: FlowEvent) -> dict[str, Any] | None:
        # Identity is capture_id + sequence; equal-valued prints are distinct.
        if event.sequence <= self.last_sequence:
            raise ValueError("DUPLICATE_OR_OUT_OF_ORDER_CAPTURE_SEQUENCE")
        self.last_sequence = event.sequence
        self.last_received_at = event.received_at
        if event.kind in {"gap", "end"}:
            self.minute_bar(event)["has_gap"] = True
            self.quote = None
            self.fields.clear()
            if not self.gaps or self.gaps[-1].get("end_at") is not None:
                if len(self.gaps) >= 64:
                    raise ValueError("COVERAGE_SEGMENT_LIMIT")
                self.gaps.append({"at": event.received_at, "reason": event.reason, "end_at": None})
            return None
        if event.kind == "resume":
            if self.gaps and self.gaps[-1].get("end_at") is None:
                self.gaps[-1]["end_at"] = event.received_at
            return None
        if event.kind == "market_data_type":
            self.market_data_type = event.tick_type or 0
            self.quote = None
            self.fields.clear()
            return None
        if event.kind == "quote":
            self.minute_bar(event)
            self.last_quote_at = event.received_at
            self.first_received_at = self.first_received_at or event.received_at
            self.quote = event
            for name in ("bid", "ask", "bid_size", "ask_size"):
                if not event.field_name or name == event.field_name:
                    self.fields[name] = (
                        getattr(event, name),
                        event.monotonic_ns,
                        event.received_at,
                    )
            return None
        if event.kind != "trade":
            return None
        self.first_received_at = self.first_received_at or event.received_at
        self.last_trade_at = event.received_at
        if event.size is None or not isfinite(event.size) or event.size < 0:
            direction, reason, coarse = "EXCLUDED", "INVALID_SIZE", False
            size = 0.0  # Retained raw; unknowable volume is never invented.
        else:
            size = event.size
            direction, reason, coarse = self.classify(event)
        bar = self.minute_bar(event)
        minute = bar["minute"]
        key = {
            "BUY_EST": VOLUMES[0],
            "SELL_EST": VOLUMES[1],
            "UNKNOWN": VOLUMES[2],
            "EXCLUDED": VOLUMES[3],
        }[direction]
        for totals in (self.totals, bar):
            totals[key] += size
            totals["trade_count"] += 1
            totals["classified_trade_count"] += int(direction in {"BUY_EST", "SELL_EST"})
        bar["coarse_timestamp_trades"] += int(coarse)
        if event.price is not None and isfinite(event.price) and event.price > 0:
            bar["price"] = event.price
        self.reasons[reason] = self.reasons.get(reason, 0) + size
        return {
            "direction": direction,
            "reason": reason,
            "coarse_timestamp_ambiguity": coarse,
            "minute": minute,
        }

    def snapshot(self) -> dict[str, Any]:
        cumulative = 0.0
        bars = []
        for key in sorted(self.bars):
            bar = metrics(self.bars[key])
            cumulative += bar["volume_delta"]
            bars.append({**bar, "cumulative_volume_delta": cumulative})
        return {
            "totals": metrics(self.totals),
            "bars": bars,
            "first_received_at": self.first_received_at,
            "last_received_at": self.last_received_at,
            "last_trade_at": self.last_trade_at,
            "last_quote_at": self.last_quote_at,
            "gaps": list(self.gaps),
            "reason_volume": dict(self.reasons),
            "last_sequence": self.last_sequence,
            "field_update_times": {k: v[2] for k, v in self.fields.items()},
        }
