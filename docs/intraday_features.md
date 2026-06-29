# Intraday Feature Layer

The intraday feature layer is research-only feature engineering for 5-minute and
other intraday bars. It does not define strategies, trade execution, broker
behavior, paper trading, or candidate promotion.

## Leakage Policy

Feature values are deterministic and are known at the current bar close unless a
column explicitly marks the value unavailable. The vectorized backtest applies
target positions with a one-bar lag, so future templates must consume these
features in a way that avoids trading on same-bar information.

Tests cover two leakage boundaries:

- Mutating rows after an evaluation window must not change feature values inside
  that evaluation window.
- Context-window feature generation must match full-frame feature generation for
  evaluation rows when enough prior context is present.

## Session Reset Policy

Session-aware features reset by `session_date`. When a market calendar is
provided, XNYS schedule opens and closes define regular-session boundaries,
including DST shifts and early closes. If a full calendar is unavailable, the
feature layer falls back to observed first and last timestamps and marks that
limitation through session warning columns.

Incomplete sessions and timestamp-grid anomalies are surfaced through:

- `session_complete_warning`
- `session_warning_reason`
- `session_missing_bar_count`
- `session_extra_bar_count`

Warnings are diagnostic flags. They are not silently ignored and they do not
manufacture candidates.

## Opening Range Availability

Opening range columns are unavailable until the configured opening window has
completed. For a 30-minute opening range on 5-minute bars, the range uses bars
from 0 through 25 minutes after the open and becomes available on the 30-minute
bar. Later bars in the session never define opening-range values before they are
known.

## VWAP Calculation

Session VWAP resets each session and uses typical price:

```text
(high + low + close) / 3
```

Missing or zero volume is handled safely. VWAP remains unavailable until
cumulative positive volume exists for the session.

## Relative Volume Policy

Relative volume compares the current bar, or cumulative volume through the
current bar, with prior sessions at the same `bar_index_in_session`. It does not
use future bars from the same session. If the configured number of prior sessions
is unavailable, relative-volume fields are `NaN`.

## Context Rows

Context rows may warm features such as previous-session levels, VWAP, opening
range, and rolling range measures. Context rows are still not scored by the
research harness. Feature generation is designed so future walk-forward windows
can build features on context plus evaluation rows without relying on rows after
the evaluation window.

## Limitations

The feature layer assumes normalized OHLCV bars with UTC timestamps. It supports
regular-session intraday research and makes incomplete sessions visible, but it
does not repair vendor data, infer missing bars, define entries or exits, or
apply trading policy. Position flattening and no-overnight scoring remain in the
research position-policy layer.
