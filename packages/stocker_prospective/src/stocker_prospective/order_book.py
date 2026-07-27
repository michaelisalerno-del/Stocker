"""Deterministic bounded IBKR depth-book reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from stocker_prospective.events import (
    DepthOperation,
    DepthRow,
    DepthSide,
    UnderlyingDepthEvent,
    UnderlyingDepthSnapshot,
)

EPSILON = 1e-12


@dataclass(frozen=True)
class DepthGap:
    symbol: str
    started_at_utc: datetime
    reason: str
    reset_number: int


@dataclass
class _MutableDepthRow:
    price: float
    size: float
    venue: str | None


class DepthBook:
    """Apply insert/update/remove callbacks without spanning reset gaps."""

    def __init__(self, *, symbol: str, con_id: int, rows_per_side: int) -> None:
        if rows_per_side <= 0:
            raise ValueError("rows_per_side must be positive")
        self.symbol = symbol
        self.con_id = con_id
        self.rows_per_side = rows_per_side
        self._bids: list[_MutableDepthRow] = []
        self._asks: list[_MutableDepthRow] = []
        self._valid = False
        self._reset_count = 0
        self._last_monotonic_ns = 0
        self._smart_depth = True
        self._previous_centroid: float | None = None
        self._bid_additions = 0.0
        self._bid_removals = 0.0
        self._ask_additions = 0.0
        self._ask_removals = 0.0

    def _record_liquidity_change(
        self,
        *,
        side: DepthSide,
        old_size: float,
        new_size: float,
    ) -> None:
        addition = max(new_size - old_size, 0.0)
        removal = max(old_size - new_size, 0.0)
        if side is DepthSide.BID:
            self._bid_additions += addition
            self._bid_removals += removal
        else:
            self._ask_additions += addition
            self._ask_removals += removal

    def apply(self, event: UnderlyingDepthEvent) -> None:
        if event.symbol != self.symbol or event.con_id != self.con_id:
            raise ValueError("depth event contract identity mismatch")
        if event.received_monotonic_ns < self._last_monotonic_ns:
            raise ValueError("depth event monotonic order regressed")
        self._last_monotonic_ns = event.received_monotonic_ns
        self._smart_depth = event.smart_depth
        if event.reset:
            self.reset(event.received_timestamp_utc, reason="ibkr_depth_reset_event")
            return
        rows = self._bids if event.side is DepthSide.BID else self._asks
        if event.operation is DepthOperation.REMOVE:
            if event.position >= len(rows):
                raise ValueError("depth remove position is absent")
            self._record_liquidity_change(
                side=event.side,
                old_size=rows[event.position].size,
                new_size=0.0,
            )
            rows.pop(event.position)
        else:
            if event.price is None or event.size is None:
                raise ValueError("depth insert/update requires price and size")
            if event.price <= 0.0 or event.size < 0.0:
                raise ValueError("depth price/size is invalid")
            row = _MutableDepthRow(
                price=event.price,
                size=event.size,
                venue=event.market_maker_or_exchange,
            )
            if event.operation is DepthOperation.INSERT:
                if event.position > len(rows):
                    raise ValueError("depth insert position has a gap")
                self._record_liquidity_change(
                    side=event.side,
                    old_size=0.0,
                    new_size=row.size,
                )
                rows.insert(event.position, row)
            else:
                if event.position >= len(rows):
                    raise ValueError("depth update position is absent")
                self._record_liquidity_change(
                    side=event.side,
                    old_size=rows[event.position].size,
                    new_size=row.size,
                )
                rows[event.position] = row
        for displaced in rows[self.rows_per_side :]:
            self._record_liquidity_change(
                side=event.side,
                old_size=displaced.size,
                new_size=0.0,
            )
        del rows[self.rows_per_side :]

    def mark_complete(self) -> None:
        if len(self._bids) < self.rows_per_side or len(self._asks) < self.rows_per_side:
            raise ValueError(
                "both depth sides require the configured rows before the book is complete"
            )
        self._valid = True
        self._bid_additions = 0.0
        self._bid_removals = 0.0
        self._ask_additions = 0.0
        self._ask_removals = 0.0

    def reset(self, timestamp: datetime, *, reason: str) -> DepthGap:
        self._bids.clear()
        self._asks.clear()
        self._valid = False
        self._reset_count += 1
        self._previous_centroid = None
        self._bid_additions = 0.0
        self._bid_removals = 0.0
        self._ask_additions = 0.0
        self._ask_removals = 0.0
        return DepthGap(
            symbol=self.symbol,
            started_at_utc=timestamp,
            reason=reason,
            reset_number=self._reset_count,
        )

    def snapshot(
        self,
        timestamp: datetime,
        *,
        advance_centroid_baseline: bool = False,
    ) -> UnderlyingDepthSnapshot:
        """Build one snapshot; only persisted snapshots may advance the centroid."""

        bid_rows = tuple(
            DepthRow(
                position=index,
                price=row.price,
                size=row.size,
                market_maker_or_exchange=row.venue,
            )
            for index, row in enumerate(self._bids)
        )
        ask_rows = tuple(
            DepthRow(
                position=index,
                price=row.price,
                size=row.size,
                market_maker_or_exchange=row.venue,
            )
            for index, row in enumerate(self._asks)
        )
        if not self._valid:
            return UnderlyingDepthSnapshot(
                snapshot_timestamp_utc=timestamp,
                received_monotonic_ns=self._last_monotonic_ns,
                session=timestamp.date(),
                symbol=self.symbol,
                con_id=self.con_id,
                bid_rows=bid_rows,
                ask_rows=ask_rows,
                total_bid_size=None,
                total_ask_size=None,
                depth_imbalance=None,
                weighted_depth_imbalance=None,
                distance_weighted_bid_liquidity=None,
                distance_weighted_ask_liquidity=None,
                bid_depth_additions=self._bid_additions,
                bid_depth_removals=self._bid_removals,
                ask_depth_additions=self._ask_additions,
                ask_depth_removals=self._ask_removals,
                bid_side_replenishment=None,
                ask_side_replenishment=None,
                depth_centroid_shift=None,
                book_slope=None,
                active_venues=0,
                near_touch_bid_liquidity=None,
                near_touch_ask_liquidity=None,
                book_valid=False,
                reset_count=self._reset_count,
                smart_depth=self._smart_depth,
            )
        total_bid = sum(row.size for row in self._bids)
        total_ask = sum(row.size for row in self._asks)
        best_bid = self._bids[0].price
        best_ask = self._asks[0].price
        weighted_bid = sum(
            row.size / (1.0 + max((best_bid - row.price) / max(best_bid, EPSILON) * 10_000.0, 0.0))
            for row in self._bids
        )
        weighted_ask = sum(
            row.size / (1.0 + max((row.price - best_ask) / max(best_ask, EPSILON) * 10_000.0, 0.0))
            for row in self._asks
        )
        all_size = total_bid + total_ask
        weighted_size = weighted_bid + weighted_ask
        centroid_numerator = sum(row.price * row.size for row in self._bids + self._asks)
        centroid = centroid_numerator / max(all_size, EPSILON)
        centroid_shift = (
            None if self._previous_centroid is None else centroid - self._previous_centroid
        )
        if advance_centroid_baseline:
            self._previous_centroid = centroid
        venues = {row.venue for row in self._bids + self._asks if row.venue is not None}
        bid_price_depth = max(best_bid - self._bids[-1].price, 0.0)
        ask_price_depth = max(self._asks[-1].price - best_ask, 0.0)
        book_slope = 0.5 * (
            bid_price_depth / max(total_bid, EPSILON) + ask_price_depth / max(total_ask, EPSILON)
        )
        return UnderlyingDepthSnapshot(
            snapshot_timestamp_utc=timestamp,
            received_monotonic_ns=self._last_monotonic_ns,
            session=timestamp.date(),
            symbol=self.symbol,
            con_id=self.con_id,
            bid_rows=bid_rows,
            ask_rows=ask_rows,
            total_bid_size=total_bid,
            total_ask_size=total_ask,
            depth_imbalance=(total_bid - total_ask) / max(all_size, EPSILON),
            weighted_depth_imbalance=(weighted_bid - weighted_ask) / max(weighted_size, EPSILON),
            distance_weighted_bid_liquidity=weighted_bid,
            distance_weighted_ask_liquidity=weighted_ask,
            bid_depth_additions=self._bid_additions,
            bid_depth_removals=self._bid_removals,
            ask_depth_additions=self._ask_additions,
            ask_depth_removals=self._ask_removals,
            bid_side_replenishment=(
                None if self._bid_removals <= EPSILON else self._bid_additions / self._bid_removals
            ),
            ask_side_replenishment=(
                None if self._ask_removals <= EPSILON else self._ask_additions / self._ask_removals
            ),
            depth_centroid_shift=centroid_shift,
            book_slope=book_slope,
            active_venues=len(venues),
            near_touch_bid_liquidity=self._bids[0].size,
            near_touch_ask_liquidity=self._asks[0].size,
            book_valid=True,
            reset_count=self._reset_count,
            smart_depth=self._smart_depth,
        )
