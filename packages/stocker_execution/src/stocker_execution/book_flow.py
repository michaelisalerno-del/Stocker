"""Fixed, causal diagnostics of sampled books. Never imported by trading rules/broker.

History comes from the recorder's retained observations, not another subscription or archive.
See docs/BOOK-FLOW.md for formulas, units and deliberately unsupported tape/volume claims.
"""

import math
from typing import Any

from stocker_execution.contracts import USABLE_PRICE_TYPES

VERSION = "SAXO_SAMPLED_BOOK_FLOW_V2"  # V2 (2026-10-01): volume change
LEVELS = (1, 3, 5, 10)
LOOKBACKS = (5, 30, 60)
FIELDS = {
    "Quote": (
        "Bid",
        "Ask",
        "BidSize",
        "AskSize",
        "DelayedByMinutes",
        "PriceTypeBid",
        "PriceTypeAsk",
        "ErrorCode",
        "MarketState",
    ),
    "PriceInfoDetails": ("LastTraded", "LastTradedSize", "Volume"),
    "MarketDepth": (
        "Bid",
        "Ask",
        "BidSize",
        "AskSize",
        "BidOrders",
        "AskOrders",
        "NoOfBids",
        "NoOfOffers",
        "UsingOrders",
    ),
}


def number(value: Any, minimum: float | None = None) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) and (minimum is None or value >= minimum) else None


def grid(price: Any, tick: float | None) -> int | None:
    p = number(price)
    if p is None or tick is None or tick <= 0:
        return None
    integer = round(p / tick)
    return integer if abs(p / tick - integer) < 1e-6 else None


def ratio(bid: float | None, ask: float | None) -> float | None:
    if bid is None or ask is None or bid + ask <= 0:
        return None
    return (bid - ask) / (bid + ask)


def side_levels(depth: dict[str, Any], side: str, tick: float | None) -> list[dict[str, Any]]:
    prices, sizes, orders = (depth.get(side + suffix) for suffix in ("", "Size", "Orders"))
    if not isinstance(prices, list):
        return []
    count = depth.get("NoOfBids" if side == "Bid" else "NoOfOffers")
    if isinstance(count, int) and not isinstance(count, bool):
        if count < 0:
            return []
        prices = prices[:count]
    result = []
    previous = None
    for i, price in enumerate(prices[:10]):
        index = grid(price, tick)
        if index is None or (
            previous is not None and (index >= previous if side == "Bid" else index <= previous)
        ):
            break
        size = number(sizes[i], 0) if isinstance(sizes, list) and i < len(sizes) else None
        count_value = number(orders[i], 0) if isinstance(orders, list) and i < len(orders) else None
        order_count = (
            count_value
            if depth.get("UsingOrders") is True
            and count_value is not None
            and count_value.is_integer()
            else None
        )
        result.append({"tick": index, "size": size, "orders": order_count})
        previous = index
    return result


def total(rows: list[dict[str, Any]], n: int, field: str) -> float | None:
    selected = rows[:n]
    if len(selected) != n or any(r[field] is None for r in selected):
        return None
    return float(sum(r[field] for r in selected))


def interval(
    history: list[dict[str, Any]], current: dict[str, Any], seconds: int
) -> list[dict[str, Any]]:
    start, end = current["at"] - seconds, current["at"]
    previous = [p for p in history if p["at"] <= start]
    if not previous:
        return []
    points = [previous[-1], *(p for p in history if start < p["at"] < end), current]
    for left, right in zip(points, points[1:], strict=False):
        if (
            left["status"] not in {"CURRENT", "DELAYED"}
            or left["delay_minutes"] != current["delay_minutes"]
            or left["generation"] != current["generation"]
            or left["valid_until"] < right["at"]
            or left["at"] > right["at"]
        ):
            return []
    return points


def temporal(history: list[dict[str, Any]], current: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for seconds in LOOKBACKS:
        points = interval(history, current, seconds)
        rows = {}
        for n in LEVELS:
            row: dict[str, Any] = {
                "status": "INSUFFICIENT_HISTORY",
                "bid_change": None,
                "ask_change": None,
                "matched_bid_change": None,
                "matched_ask_change": None,
            }
            if points and all(p["depth"][str(n)]["imbalance"] is not None for p in points):
                before, after = points[0]["depth"][str(n)], current["depth"][str(n)]
                row.update(
                    status="AVAILABLE",
                    bid_change=after["bid"] - before["bid"],
                    ask_change=after["ask"] - before["ask"],
                )
                for side in ("bid", "ask"):
                    old = dict(points[0]["basis"][side][:n])
                    new = dict(current["basis"][side][:n])
                    common = old.keys() & new.keys()
                    row[f"matched_{side}_change"] = (
                        sum(new[k] - old[k] for k in common) if common else None
                    )
                    row[f"matched_{side}_levels"] = len(common)
                if seconds == 60:
                    durations = {"bid_heavy": 0.0, "ask_heavy": 0.0, "balanced": 0.0}
                    for left, right in zip(points, points[1:], strict=False):
                        value = left["depth"][str(n)]["imbalance"]
                        label = (
                            "bid_heavy" if value > 0 else "ask_heavy" if value < 0 else "balanced"
                        )
                        durations[label] += right["at"] - max(left["at"], current["at"] - seconds)
                    row.update(
                        {
                            label + "_fraction": duration / seconds
                            for label, duration in durations.items()
                        }
                    )
            rows[str(n)] = row
        result[str(seconds)] = rows
    return result


def observe(
    identity: dict[str, Any],
    value: dict[str, Any] | None,
    at: float,
    context: dict[str, Any],
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    value = value or {}
    groups = {name: value[name] if isinstance(value.get(name), dict) else {} for name in FIELDS}
    quote, details, depth = (groups[name] for name in ("Quote", "PriceInfoDetails", "MarketDepth"))
    tick = number(identity.get("tick_size"), 0)
    delay = number(quote.get("DelayedByMinutes"), 0)
    bid, ask = grid(quote.get("Bid"), tick), grid(quote.get("Ask"), tick)
    quality = [
        "INVALID_" + name.upper() + "_SHAPE"
        for name in FIELDS
        if value.get(name) is not None and not isinstance(value[name], dict)
    ]
    if not value or context.get("problem") or context.get("valid_until", at - 1) < at:
        quality.append("RECONSTRUCTION_OR_STREAM_UNAVAILABLE")
    if delay is None:
        quality.append("DELAY_UNKNOWN")
    if bid is None or ask is None or bid > ask:
        quality.append("INVALID_QUOTE_OR_TICK_GRID")
    if quote.get("ErrorCode") not in (None, "None") or any(
        quote.get(k) not in USABLE_PRICE_TYPES for k in ("PriceTypeBid", "PriceTypeAsk")
    ):
        quality.append("QUOTE_STATUS_UNAVAILABLE")
    bids, asks = side_levels(depth, "Bid", tick), side_levels(depth, "Ask", tick)
    if bids and asks and bids[0]["tick"] > asks[0]["tick"]:
        quality.append("CROSSED_DEPTH")
        bids, asks = [], []
    status = "UNAVAILABLE" if quality else "DELAYED" if delay else "CURRENT"
    quote_usable = status != "UNAVAILABLE"
    if not bids or not asks:
        quality.append("L2_UNAVAILABLE")
    available = {
        group: {field: groups[group].get(field) is not None for field in fields}
        for group, fields in FIELDS.items()
    }
    result: dict[str, Any] = {
        "version": VERSION,
        "at": at,
        "status": status,
        "quality_flags": quality,
        "generation": context.get("subscription_id", ""),
        "valid_until": context.get("valid_until", at),
        "delay_minutes": delay,
        "available_fields": available,
        "depth": {},
        "basis": {
            "bid": [[r["tick"], r["size"]] for r in bids],
            "ask": [[r["tick"], r["size"]] for r in asks],
        },
        "available_levels": {"bid": len(bids), "ask": len(asks)},
        "spread_ticks": ask - bid if quote_usable and ask is not None and bid is not None else None,
        "weighted_midpoint": None,
        "weighted_displacement_ticks": None,
        "latest_trade": {
            "price": number(details.get("LastTraded")),
            "size": number(details.get("LastTradedSize"), 0),
        },
        # Verified on LIVE 2026-10-01: Saxo's Volume is the contracts traded since the 18:00 New
        # York session open (the summed one-minute chart volumes), so a change is traded volume.
        "volume": {
            "value": number(details.get("Volume"), 0),
            "change": None,
            "changes": {},
            "status": "SESSION_CUMULATIVE",
        },
        "feed": context,
    }
    previous = history[-1] if history else None
    if previous and previous["at"] <= at and previous["generation"] == result["generation"]:
        old_volume, volume = previous["volume"]["value"], result["volume"]["value"]
        if old_volume is not None and volume is not None and volume < old_volume:
            result["volume"]["status"] = "RESET_OR_CORRECTION_UNCLASSIFIED"
    for n in LEVELS:
        b, a = (total(rows, n, "size") for rows in (bids, asks))
        bo, ao = (total(rows, n, "orders") for rows in (bids, asks))
        usable = status != "UNAVAILABLE"
        imbalance = ratio(b, a) if usable else None
        result["depth"][str(n)] = {
            "bid": b if usable else None,
            "ask": a if usable else None,
            "imbalance": imbalance,
            "order_imbalance": ratio(bo, ao) if usable else None,
            "label": "UNAVAILABLE"
            if imbalance is None
            else "BID_HEAVY"
            if imbalance > 0
            else "ASK_HEAVY"
            if imbalance < 0
            else "BALANCED",
        }
    bs, az = number(quote.get("BidSize"), 0), number(quote.get("AskSize"), 0)
    if (
        status != "UNAVAILABLE"
        and bs is not None
        and az is not None
        and bs + az > 0
        and bid is not None
        and ask is not None
        and tick
    ):
        weighted = (ask * bs + bid * az) / (bs + az)
        result.update(
            weighted_midpoint=weighted * tick,
            weighted_displacement_ticks=weighted - (bid + ask) / 2,
        )
    result["lookbacks"] = (
        temporal(history, result) if status != "UNAVAILABLE" else temporal([], result)
    )
    if result["volume"]["status"] == "SESSION_CUMULATIVE" and status != "UNAVAILABLE":
        for seconds in LOOKBACKS:
            points = interval(history, result, seconds)
            values = [p["volume"]["value"] for p in points]
            # Any fall inside the window is a session reset or a correction: no change then.
            if (
                values
                and None not in values
                and all(a <= b for a, b in zip(values, values[1:], strict=False))
            ):
                result["volume"]["changes"][str(seconds)] = values[-1] - values[0]
        result["volume"]["change"] = result["volume"]["changes"].get("60")
    return result
