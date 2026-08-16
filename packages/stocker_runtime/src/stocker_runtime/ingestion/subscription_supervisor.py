"""Small retry policy for independent IBKR market-data subscriptions."""

from __future__ import annotations

from dataclasses import dataclass

MAX_RETRY_DELAY_US = 60_000_000
_PERMANENT_REJECTION_CODES = frozenset({200, 354, 10167, 10168})


@dataclass(frozen=True)
class SubscriptionRetry:
    """Persisted disposition for one failed subscription attempt."""

    permanent: bool
    next_retry_at_us: int | None


def retry_disposition(
    *,
    kind: str,
    code: int | None,
    retry_count: int,
    failed_at_us: int,
) -> SubscriptionRetry:
    """Classify only the typed statuses emitted by the official bridge."""

    if retry_count < 1 or failed_at_us < 0:
        raise ValueError("subscription retry count and failure time are invalid")
    permanent = kind == "request_rejected" and (code in _PERMANENT_REJECTION_CODES or code is None)
    if permanent:
        return SubscriptionRetry(permanent=True, next_retry_at_us=None)
    base_delay_us = 5_000_000 if kind == "pacing" else 1_000_000
    exponent = min(retry_count - 1, 6)
    delay_us = min(MAX_RETRY_DELAY_US, base_delay_us * 2**exponent)
    return SubscriptionRetry(
        permanent=False,
        next_retry_at_us=failed_at_us + delay_us,
    )
