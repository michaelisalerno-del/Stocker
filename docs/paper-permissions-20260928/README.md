# PAPER permissions test — 2026-09-28

The user reported that live futures/options approval was not yet granted and requested a test.
The verified PAPER account was `DUP655399`; no live endpoint/account was contacted.

## Observed results

| Diagnostic sample | Quote | Non-executing What-if result |
|---|---|---|
| BTCV6 future, conId 876880607 | Fresh real-time bid/ask | Error 201: No Trading Permission, Customer Ineligible |
| BTCV6 C85000 monthly call, conId 877338421 | Fresh real-time bid/ask | Same error 201 |
| NQZ6 future, conId 563947726 | Fresh real-time bid/ask | Margin/commission estimate returned, no permission rejection |
| Q4AU6 P30700, September 28 NQ put, conId 922982859 | Fresh real-time bid/ask | Margin/commission estimate returned, no permission rejection |

BTC's broker rejection text includes: “Retail clients from your country cannot open positions
in Crypto ETPs that do not have a UK listing.” That is IBKR's wording; the qualified contracts
are FUT/FOP, not ETPs. It is evidence of an account/product eligibility rejection, not a legal
determination. IBKR must clarify why this message applies to these CME contracts and whether
the account can obtain permission. No country/classification or permission setting was changed.

NQ preview acceptance is not proof of execution permission in every circumstance or a future fill.
The returned `PreSubmitted` belongs to a What-if response and is **not a working order**.
This was not a general approval for every future/option or a test of the full frozen strategy.

## Method and boundaries

Used IBKR's documented `whatIf=True` commission/margin preview on the PAPER account.
[IBKR explains](https://interactivebrokers.github.io/tws-api/margin.html) that this requests a
credit check instead of routing an order to a destination; the
[current Order reference](https://www.interactivebrokers.com/docs/tws-api/ref/order) exposes that
flag for commission/margin information. `transmit=False` parked orders were not used.

Both the high-level executable-order methods and low-level wire access were guarded.
The low-level placeOrder boundary accepted only `whatIf is True`, the allowlisted PAPER account,
one BUY LMT contract and the explicitly qualified diagnostic identities. Order cancellations
and global cancellation were forbidden. Before/after broker snapshots returned no open orders,
positions or executions. **Zero executable orders were sent.**

The BTC call was the already-qualified monthly entitlement sample, not an approved replacement
for the frozen 0DTE rule. The NQ put was found through real returned chain identity, earliest
available expiry and a diagnostic strike near the current future; underlying linkage and product
fields were verified. It was not frozen-delta selection and did not approve a product mapping,
fee reserve, strike tolerance or affordability. No direct-futures strategy was introduced.

Reused the existing BrokerConnection, request scheduler and reference-counted subscription
manager in temporary client 89. First probe: three quote lines; second: two. Each returned to
zero. Wire cap five requests/second, queue high water one, no pacing rejections. No history,
depth, tick-by-tick or full-chain quote collection; only option-chain metadata was requested.
The diagnostic scripts were archived under the server backup's `debug` directory.

IBKR's [PAPER account documentation](https://www.interactivebrokers.com/campus/trading-lessons/request-paper-trading-account/)
says paper permissions match the regular live account. Market-data access alone therefore cannot
establish order permission. The actual mixed responses above are reported separately from that policy.

## Final application state

[Postflight at 04:32:00 UTC / 05:32:00 London](postflight.json): code release
`3ac8cf32afcc78d602bf1041d79b72fa97e99212`, connected/reconciled, PAPER arming still enabled,
LIVE unavailable, no ledger orders/fills/reservations/positions. Thirteen app-owned lines,
zero temporary quotes or depth. All six underlyings reported current bars.

All entries remain blocked by unverified data allowance and missing approved option mappings;
NQ additionally lacks required prior-volume reference history. BTC has the demonstrated broker
permission rejection as a further unresolved constraint. Nothing was deployed, restarted, purchased
or enabled by this test. No config or strategy changes were made.

## Evidence

- [BTC futures/option and NQ future preview responses](btc-and-nq-future.json), 04:28:37–04:28:46 UTC.
- [NQ option discovery and preview responses](nq-option.json), 04:31:10–04:31:18 UTC.
- [Application postflight](postflight.json).
- Protected server originals and probes: `/var/lib/stocker/backups/paper-permissions-20260928`.

Preview commission values are estimates, not reported execution commissions. Raw unset numeric
sentinels in those broker responses are retained as returned and must not be treated as costs.
