"""Exact-timestamp checkpoint selection and frozen directional acceptance."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

BAR_DURATION = pd.Timedelta(minutes=5)


def _utc(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(str(value))
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


@dataclass(frozen=True)
class CheckpointBar:
    """The one and only registered first completed post-anchor bar."""

    status: str
    bar_start_timestamp: pd.Timestamp | None = None
    freeze_timestamp: pd.Timestamp | None = None
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float | None = None


@dataclass(frozen=True)
class PriceAcceptance:
    """Direction-adjusted values frozen when the checkpoint bar closes."""

    status: str
    signed_close_return_bps: float | None = None
    favourable_excursion_bps: float | None = None
    adverse_excursion_bps: float | None = None
    acceptance_balance_bps: float | None = None
    price_acceptance_pass: bool = False


def select_first_post_anchor_bar(
    bars: pd.DataFrame,
    *,
    anchor_timestamp: pd.Timestamp,
) -> CheckpointBar:
    """Select exactly ``anchor + 5 minutes``; never substitute a later row."""

    required = {"timestamp", "open", "high", "low", "close"}
    if bars.empty or required - set(bars.columns):
        return CheckpointBar(status="missing_source_data")
    expected = _utc(anchor_timestamp) + BAR_DURATION
    timestamps = pd.to_datetime(bars["timestamp"], utc=True, errors="coerce")
    matched = bars.loc[timestamps.eq(expected)]
    if matched.empty:
        return CheckpointBar(status="missing_first_post_anchor_bar")
    if len(matched) != 1:
        return CheckpointBar(status="ambiguous_first_post_anchor_bar")
    row = matched.iloc[0]
    values = pd.to_numeric(row[["open", "high", "low", "close"]], errors="coerce")
    if values.isna().any() or float(values["low"]) > float(values["high"]):
        return CheckpointBar(status="invalid_checkpoint_ohlc")
    return CheckpointBar(
        status="available",
        bar_start_timestamp=expected,
        freeze_timestamp=expected + BAR_DURATION,
        open=float(values["open"]),
        high=float(values["high"]),
        low=float(values["low"]),
        close=float(values["close"]),
    )


def calculate_price_acceptance(
    checkpoint: CheckpointBar,
    *,
    anchor_reference_price: float,
    direction: int,
) -> PriceAcceptance:
    """Apply the registered sign-and-excursion rule without fitted parameters."""

    if checkpoint.status != "available":
        return PriceAcceptance(status=checkpoint.status)
    if direction not in (-1, 1):
        return PriceAcceptance(status="ambiguous_direction")
    reference = float(anchor_reference_price)
    if reference <= 0.0:
        return PriceAcceptance(status="invalid_anchor_reference_price")
    if checkpoint.high is None or checkpoint.low is None or checkpoint.close is None:
        return PriceAcceptance(status="invalid_checkpoint_ohlc")
    if direction == 1:
        close = 10_000.0 * (checkpoint.close / reference - 1.0)
        favourable = 10_000.0 * (checkpoint.high / reference - 1.0)
        adverse = 10_000.0 * (1.0 - checkpoint.low / reference)
    else:
        close = 10_000.0 * (1.0 - checkpoint.close / reference)
        favourable = 10_000.0 * (1.0 - checkpoint.low / reference)
        adverse = 10_000.0 * (checkpoint.high / reference - 1.0)
    balance = favourable - adverse
    return PriceAcceptance(
        status="available",
        signed_close_return_bps=close,
        favourable_excursion_bps=favourable,
        adverse_excursion_bps=adverse,
        acceptance_balance_bps=balance,
        price_acceptance_pass=bool(close > 0.0 and favourable > adverse),
    )
