"""Bounded binary framing and Saxo price-field reconstruction, independent of transport."""

import copy
import hashlib
import json
import struct
from collections import OrderedDict, deque
from typing import Any

from stocker_execution.book_flow import FIELDS


def merge(previous: Any, update: Any) -> Any:
    """Missing properties survive; explicit null clears; price arrays replace in full.

    Options-board indexed arrays and chart samples use their own reducers, not this function.
    """
    if not isinstance(update, dict):
        return copy.deepcopy(update)
    result = copy.deepcopy(previous) if isinstance(previous, dict) else {}
    for key, value in update.items():
        result[key] = merge(result.get(key), value)
    # Explicit zero depth counts also invalidate any prior price/size arrays.
    if isinstance(result.get("MarketDepth"), dict):
        depth = result["MarketDepth"]
        for count, side in (("NoOfBids", "Bid"), ("NoOfOffers", "Ask")):
            if depth.get(count) == 0:
                for suffix in ("", "Size", "Orders"):
                    depth[side + suffix] = []
    return result


class Frames:
    def __init__(self, limit: int = 256 * 1024):
        self.buffer = bytearray()
        self.limit = limit

    def feed(self, data: bytes) -> list[dict[str, Any]]:
        if len(self.buffer) + len(data) > self.limit:
            self.buffer.clear()
            raise ValueError("STREAM_FRAME_LIMIT")
        self.buffer.extend(data)
        result = []
        while len(self.buffer) >= 16:
            reference_size = self.buffer[10]
            header = 16 + reference_size
            if len(self.buffer) < header:
                break
            size = struct.unpack_from("<I", self.buffer, 12 + reference_size)[0]
            if size + header > self.limit:
                self.buffer.clear()
                raise ValueError("STREAM_PAYLOAD_LIMIT")
            if len(self.buffer) < header + size:
                break
            if self.buffer[11 + reference_size] != 0:
                self.buffer.clear()
                raise ValueError("UNSUPPORTED_STREAM_FORMAT")
            reference = bytes(self.buffer[11 : 11 + reference_size]).decode("ascii")
            message_id = str(struct.unpack_from("<Q", self.buffer)[0])
            raw = bytes(self.buffer[header : header + size])
            payload = json.loads(raw)
            if not isinstance(payload, (dict, list)):
                raise ValueError("INVALID_STREAM_PAYLOAD")
            result.append({"message_id": message_id, "reference": reference, "payload": payload})
            del self.buffer[: header + size]
        return result


class PriceState:
    def __init__(self) -> None:
        self.value: dict[str, Any] | None = None
        self.seen: OrderedDict[str, bytes] = OrderedDict()
        self.last_message_id: str | None = None
        self.generation = ""
        self.receipt: float | None = None
        self.quote_times: dict[str, float] = {}
        self.depth_receipt: float | None = None
        self.size_receipt: float | None = None
        self.size_times: dict[str, float] = {}
        self.last_contact: float | None = None
        self.problem = "AWAITING_SNAPSHOT"
        self.refresh_ms: int | None = None
        self.inactivity_timeout = 5
        self.last_receipt: float | None = None
        self.last_field_change: float | None = None
        self.field_changes: dict[str, float] = {}
        self.intervals: deque[float] = deque(maxlen=60)

    def snapshot(self, value: dict[str, Any], generation: str, at: float) -> None:
        self.value = merge({}, value)
        self.generation = generation
        self.seen.clear()
        self.quote_times.clear()
        self.size_times.clear()
        self.receipt = self.depth_receipt = self.size_receipt = None
        self.last_contact = at
        self.last_receipt = at
        self.last_field_change = at
        self.field_changes.clear()
        for group, fields in FIELDS.items():
            if not isinstance(value.get(group), dict):
                continue
            for field in fields:
                if (value.get(group) or {}).get(field) is not None:
                    self.field_changes[group + "." + field] = at
        self.intervals.clear()
        self.touch(value, at)
        self.problem = ""

    def touch(self, value: dict[str, Any], at: float) -> None:
        for side in ("Bid", "Ask"):
            if side in (value.get("Quote") or {}):
                self.quote_times[side] = at
        if len(self.quote_times) == 2:
            self.receipt = min(self.quote_times.values())
        if "MarketDepth" in value:
            self.depth_receipt = at
        for side in ("Bid", "Ask"):
            field = side + "Size"
            current_quote = (self.value or {}).get("Quote") or {}
            if field in (value.get("Quote") or {}) or (
                field not in current_quote and field in (value.get("PriceInfoDetails") or {})
            ):
                self.size_times[side] = at
        if len(self.size_times) == 2:
            self.size_receipt = min(self.size_times.values())

    def update(self, value: dict[str, Any], message_id: str, at: float) -> bool:
        self.last_contact = at
        if self.last_receipt is not None and at > self.last_receipt:
            self.intervals.append((at - self.last_receipt) * 1000)
        self.last_receipt = at
        digest = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).digest()
        if message_id in self.seen:
            if self.seen[message_id] != digest:
                self.gap("REPLAY_CONTENT_CONFLICT")
                raise ValueError("REPLAY_CONTENT_CONFLICT")
            return False
        self.seen[message_id] = digest
        if len(self.seen) > 2048:
            self.seen.popitem(last=False)
        self.last_message_id = message_id  # opaque: never subtract or order IDs
        if self.value is None:
            return False
        merged = merge(self.value, value)
        for group, fields in FIELDS.items():
            prior, current = self.value.get(group) or {}, merged.get(group) or {}
            if isinstance(prior, dict) and isinstance(current, dict):
                for field in fields:
                    if current.get(field) != prior.get(field):
                        self.field_changes[group + "." + field] = at
                        self.last_field_change = at
        self.value = merged
        self.touch(value, at)
        return True

    def gap(self, reason: str) -> None:
        self.value = None
        self.receipt = self.depth_receipt = self.size_receipt = None
        self.quote_times.clear()
        self.size_times.clear()
        self.problem = reason

    def sizes(self) -> dict[str, Any]:
        value = self.value or {}
        quote, legacy = value.get("Quote") or {}, value.get("PriceInfoDetails") or {}
        return {
            side: quote[side + "Size"] if side + "Size" in quote else legacy.get(side + "Size")
            for side in ("Bid", "Ask")
        }

    def observation_context(self) -> dict[str, Any]:
        return {
            "subscription_id": self.generation,
            "granted_refresh_ms": self.refresh_ms,
            "observed_receipt_ms": {
                "samples": len(self.intervals),
                "mean": sum(self.intervals) / len(self.intervals) if self.intervals else None,
                "minimum": min(self.intervals) if self.intervals else None,
                "maximum": max(self.intervals) if self.intervals else None,
            },
            "last_receipt": self.last_receipt,
            "last_field_change": self.last_field_change,
            "last_depth_change": max(
                (at for k, at in self.field_changes.items() if k.startswith("MarketDepth.")),
                default=None,
            ),
            "last_trade_observation_change": max(
                (
                    self.field_changes[k]
                    for k in ("PriceInfoDetails.LastTraded", "PriceInfoDetails.LastTradedSize")
                    if k in self.field_changes
                ),
                default=None,
            ),
            "last_contact": self.last_contact,
            "valid_until": (self.last_contact or 0) + self.inactivity_timeout,
            "problem": self.problem,
        }

    def depth(self, at: float) -> dict[str, Any]:
        value = self.value or {}
        raw = value.get("MarketDepth")
        depth = raw if isinstance(raw, dict) else {}
        fresh = (
            self.depth_receipt is not None
            and self.last_contact is not None
            and 0 <= at - self.last_contact <= self.inactivity_timeout
            and not self.problem
        )
        result: dict[str, Any] = {
            "status": "L1_ONLY" if value else "L2_UNAVAILABLE",
            "fresh": fresh,
            "fields": sorted(depth),
            "bids": [],
            "asks": [],
            "observation_only": True,
            "valid_until": (self.last_contact or 0) + self.inactivity_timeout,
            "last_field_change": self.last_field_change,
            "diagnostic": "Sampled provider depth; no inferred cancellations or aggressor trades",
        }
        for side, name, count in (("Bid", "bids", "NoOfBids"), ("Ask", "asks", "NoOfOffers")):
            prices = depth.get(side) or []
            sizes, orders = depth.get(side + "Size") or [], depth.get(side + "Orders") or []
            count_value = depth.get(count)
            n = min(len(prices), count_value) if isinstance(count_value, int) else len(prices)
            if fresh:
                result[name] = [
                    {
                        "price": prices[i],
                        "size": sizes[i] if i < len(sizes) else None,
                        "orders": orders[i] if i < len(orders) else None,
                    }
                    for i in range(n)
                ]
        if any(result[k] for k in ("bids", "asks")):
            result["status"] = "L2_AVAILABLE"
        elif depth and not fresh:
            result["status"] = "L2_UNAVAILABLE"
        return result


def merge_board(previous: Any, update: Any) -> Any:
    """Option-board Expiries/Strikes are keyed by Index; price arrays are replacements."""
    if not isinstance(update, dict):
        return copy.deepcopy(update)
    result = copy.deepcopy(previous) if isinstance(previous, dict) else {}
    for name, value in update.items():
        if name in {"Expiries", "Strikes"} and isinstance(value, list) and value:
            indexed = {row["Index"]: row for row in result.get(name, []) or []}
            for row in value:
                if not isinstance(row, dict) or not isinstance(row.get("Index"), int):
                    raise ValueError("OPTION_BOARD_INDEX_MISSING")
                indexed[row["Index"]] = merge_board(indexed.get(row["Index"]), row)
            if len(indexed) > 1000:
                raise ValueError("OPTION_BOARD_LIMIT")
            result[name] = [indexed[k] for k in sorted(indexed)]
        else:
            result[name] = merge_board(result.get(name), value)
    return result
