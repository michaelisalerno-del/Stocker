# BTC non-transmitting option audit — 2026-09-27

**Result: broker quote access works, but BTC is not ready to arm.** This was a test,
not a listed-product approval or a change to strategy/execution configuration.

Deployed code: `9b0466347820939215d83003470b05d0609bc389`.
Verified PAPER account: `DUP655399`. Tests ran between 16:46 and 17:01 UTC.
No orders, what-if orders, cancellations of orders, permission changes, service restarts
or subscription purchases were made. Micro/BFF products were not substituted or authorised.

**Follow-up:** the chain-ID defect was corrected and deployed in `3ac8cf3`.
See the [fix, tests and later broker evidence](../btc-chain-fix-20260927/README.md).
The results below describe the original audit before that fix; its raw records are preserved.

## Results

| Check | Result |
|---|---|
| Account / reconciliation | PASS; no open orders or nonzero positions; zero returned executions before and after |
| BTC monitoring | PASS; five reference sessions and current BTCV6 bars |
| Option metadata | PASS for actual `BTCV6 C85000`, conId 877338421, underlying 876880607, CME / BTC / USD / multiplier 5, price magnifier 1 |
| Option quote access | PASS for that monthly diagnostic sample: real-time data type 1, bid 3465 / ask 3640 at 17:01:06 UTC; positive uncrossed prices, both sides received less than five seconds earlier |
| Greeks | Bid/ask-derived and broker-model Greeks returned separately; not used to select a trade |
| Same-day option | No verified match for Sunday September 27; the returned standard chain advertised October 30 only |
| Specific Monday contracts | Five September 28 calls around the current future price returned IBKR error 200; this does not prove every weekly contract is unavailable |
| Chain matching | FAIL: the installed API decoder returns underlyingConId as a string, so the deployed integer comparison discards the otherwise matching monthly chain |
| FX and £10 budget | UNVERIFIED: GBP/USD bid and ask were -1; no approved product fee reserve, delta tolerance or frozen-selected eligible option exists |
| Expiry-time interpretation | UNRESOLVED: sampled option reports October 30, 10:00:00, US/Central; current verifier interprets that as 15:00 London, whereas the frozen BTC product mapping requires 16:00 London. The related future reports 11:00:00. Resolve field semantics with IBKR before approval |
| Cleanup | PASS; zero audit-owned subscriptions after every run; app returned to 13 lines, zero temporary quotes |
| Existing offline tests | 53 passed: futures, market-data and BTC-history suites |
| Added diagnostic reproduction | FAIL reproduced locally through the installed decoder, with no connection; string ID discarded by the deployed comparison |

The monthly option is **not 0DTE, not the frozen 0.10-delta selection, and not a proposed trade**.
Its quote demonstrates entitlement for this one contract, not every product or expiry.
Ask 3640 × multiplier 5 is USD 18,200 before fees; this diagnostic sample cannot establish
the cost of the correctly selected frozen option. No stale FX or synthetic premium was substituted.

IBKR's public low-volume [BRR commission schedule](https://www.interactivebrokers.com/en/pricing/commissions-futures.php)
lists USD 5 per contract plus fees. Its [CME fee page](https://www.interactivebrokers.com/en/accounts/fees/CME.php)
lists another USD 5 for Bitcoin futures options. These are public schedule observations, not a
verification of this account's exact tariff or an approved GBP fee reserve.
The [CME standard option rules](https://www.cmegroup.com/content/dam/cmegroup/rulebook/CME/IV/350/350A.pdf)
distinguish monthly termination with the underlying from weekly 16:00 London termination.
No cutoff discrepancy was overridden.

## Diagnostic detail and limitations

The test used temporary read-only client 89 with the existing BrokerConnection, wire pacer and
Subscriptions implementation. It capped outbound requests at five/second and used no historical
requests, depth, full-chain quote streams or live-account connection. Maximum simultaneous
probe subscriptions observed: three (one FUT, one FOP, one FX). The probe is not an additional
runtime service or a workaround for shared account allowances. External usage and the account-wide
100-line allowance remain unverified. Queue high water was one; no pacing rejection was recorded.

The first filtered chain result looked empty. Capturing the raw callback showed one chain:
`underlyingConId="876880607"`, exchange CME, class BTC, multiplier 5. The installed ib_async
2.1.0 decoder forwards this field without converting it to an integer; the app compares it to
integer 876880607 in `PaperBroker.prepare`. The [offline reproducer](chain_id_reproducer.py)
exercises that decoder and exited with an assertion failure on the audited release. It now
passes after the correction; permanent regressions are in the futures and market-data tests.

An exploratory request with an empty futures exchange returned error 321 and timed out; it was
not repeated. A later audit script stopped while serialising a market-rule named tuple. Both
incomplete runs are retained below; the final quote-access run corrected the diagnostic serializer
and completed. Neither failure is evidence of missing option permissions.

## Evidence

- [Initial filtered-chain / underlying / FX audit](btc-option-audit-20260927.json)
- [Raw chain response and rejected empty-exchange request](btc-option-direct-audit-20260927.json)
- [Specific contract definitions and incomplete quote probe](btc-option-direct-quotes-20260927.json)
- [Completed quote/Greek/FX audit and before/after exposure](btc-option-quote-access-20260927.json)
- [Post-audit app status](postflight.json)

Callback times are local receipt timestamps, not exchange timestamps. Some resource snapshots
hold references to consumer dictionaries that were emptied during cleanup; counts describe the
pre-cleanup snapshot, while those dictionaries reflect the later cleanup.

## Follow-up required at the original audit

Correct and regression-test the ID conversion at the broker boundary while retaining strict
identity checks. Resolve weekly contract discovery and expiry field semantics using exact
broker-qualified identities. Obtain usable current FX and establish the account-specific fee
reserve. Only then can a concrete product/delta-tolerance mapping be reviewed under the original
requirements. Keep unsupported clocks and unaffordable contracts blocked.

The server remains unchanged, PAPER/unarmed, with L2 disabled. Postflight at 17:01:58 UTC:
same PID 1437618, zero restarts, zero ledger orders/fills/reservations, BTC reference count five.
