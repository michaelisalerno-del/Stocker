# Entry execution protection

The HRB paper trade on 4 September 2026 used an intrabar short trigger near
$49.95 but filled near $49.66, while its stop and target remained $50.10/$49.65.
Historical qualification alone must not authorize execution at an obsolete price.

The historical HRB reproduction below retains completed-bar timing for provenance.
Current Session HARD uses actual-event timestamps, with a 60-second signal age limit and no
completed-minute extension. LONG protection mirrors SHORT; the method supplies stop/target and
a T0+15 broker timeout. The current package does not apply the legacy pooled payoff hurdle.

The shared Stage 7 checks include:

- The signal must be time-aware, not future-dated, within the original five-minute
  entry window, and less than 60 seconds beyond its completed entry bar. Gap-at-open
  signals retain their original timestamps; their freshness allowance accounts for
  the one-minute bar completing before production observes it.
- Obtain a correctly identified live bid/ask, no more than five seconds old. Frozen,
  delayed, missing, crossed, future-dated or invalid quotes reject the opportunity.
  The temporary stock stream waits at most three seconds and is always released.
- For a short, the live bid must be at least the original entry reference rounded
  **up** to the stock's tick. The ask must remain below the original stop. This
  permits no adverse entry slippage against the planned risk geometry.
- Recheck signal time after the quote await. Send a SELL limit parent at the minimum
  acceptable entry, with GTD expiry at most five seconds later, capped by both the
  freshness deadline and original entry-window deadline. Stop/target children retain
  their existing GTC protection. A price move after the quote cannot fill the short
  below its limit. An unfilled remainder must not remain a DAY market entry.

Sizing, original exit prices, qualification and pooled-payoff calculations remain
unchanged. This intentionally rejects some historically qualified opportunities;
it does not promise to reproduce research fills or improve strategy expectancy.
The historical payoff pool continues to track its existing baseline opportunities,
including those that the new execution checks reject.

The new rejection codes are `STALE_SIGNAL`, `ENTRY_QUOTE_UNAVAILABLE` and
`ENTRY_PRICE_MOVED`. Existing execution-attempt records retain the rejection detail.
Accepted plans persist `entry_limit_price` and `entry_expires_at`; additive SQLite
migration leaves historical records null. Partially filled expired entries retain
their open exposure and duplicate-order protection in the ledger.

Waiting entries are checked after required broker reconciliation and before bulk
session/preparation work, then again after checkpoint evaluation. Time-sensitive
checks reread the clock after awaits. Slow operations emit `runtime_stage_timing`;
historical requests emit `ibkr_history_timing` with separate queue/request durations.
Only operations lasting at least one second produce these timing records.

This does not establish which operation caused HRB's historical 63-second gap:
that run lacked the necessary timing records. It prevents an obsolete entry being
submitted after such a delay and makes future slow operations attributable.

## Validation

`tests/test_entry_execution_guard.py` reproduces HRB through the actual strategy
and Stage 7 with a fake broker, and verifies price/freshness rejection, bounded valid
entry, ledger persistence, and partial-fill expiry. Broker-adapter tests inspect
the transmitted GTD parent and unchanged GTC children, reject expired plans, and
verify temporary quote-stream cleanup. Runtime tests verify entry priority, fresh
clocks and rejection of obsolete restored signals. PAPER/LIVE routing and the HV
payoff tests remain applicable.

IBKR's [Order reference](https://www.interactivebrokers.com/docs/tws-api/ref/order)
documents GTD/`goodTillDate`; its
[bracket-order documentation](https://www.interactivebrokers.com/docs/general/order-types/complex-orders/bracket-orders)
documents the parent/child transmit sequence. The adapter uses UTC
`YYYYMMDD-HH:mm:ss` for the entry expiry. No test orders are required for deployment;
actual venue acceptance and fills must be observed during normal paper operation.
