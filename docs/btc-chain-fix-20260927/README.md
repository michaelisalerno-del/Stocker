# BTC chain identity fix — 2026-09-27

**Implemented and deployed; PAPER remains unarmed.** Runtime release:
`3ac8cf32afcc78d602bf1041d79b72fa97e99212`.

## Correction

The installed ib_async 2.1.0 decoder delivered option-chain underlying IDs as text.
Strict comparison with the qualified integer futures conId therefore discarded valid
chains. The existing broker wrapper now validates and converts the ID at the callback
boundary. Invalid IDs fail the request explicitly; late callbacks cannot revive a failed
or completed request. Selection still requires the exact qualified underlying identity.

No frozen rules, product mappings, delta tolerances, expiry clocks, risk limits, order
transmission settings or GC diagnostics changed. No UI changes were needed.

## Verification

Seven focused regressions failed before the correction and passed afterward, including
the real decoder path through option preparation, invalid IDs and late callbacks. The
[original reproducer](../btc-option-audit-20260927/chain_id_reproducer.py) now passes with
integer 876880607 and one matching chain.

The following suites passed **76 tests**:

```sh
rtk .venv/bin/pytest tests/test_btc_history.py tests/test_futures.py tests/test_market_data.py tests/test_futures_deployment.py tests/test_dashboard_security.py tests/test_ci_smoke.py
```

Ruff, Mypy over the execution package, diff checks and the installed server-only offline
startup/assets check passed. One existing Starlette/httpx deprecation warning remains.

A bounded, non-transmitting broker probe ran against the staged correction at
17:27:34–17:27:40 UTC using the allowlisted PAPER account. It confirmed that the actual
CME chain for future 876880607 now reaches the application with an integer ID and matches.
The probe used five requests/second, no streaming quotes, depth or history requests;
observed queue high water was one. Before/after exposure was empty. See
[the broker response and raw expiry fields](verification.json).

## Remaining BTC blocks

- The tested weekly class `P4A` returned error 200. A separate metadata search for one
  standard-BTC call strike returned eleven monthly records, including one already expired
  contract. This bounded search did not establish a valid same-day/weekly mapping and does
  not prove that every weekly listing is unavailable. Expired contracts remain ineligible.
- Raw broker expiry fields were `20261030 11:00:00 US/Central` for BTCV6 and
  `20261030 10:00:00 US/Central` for its monthly call. The timezone was present before
  decoding: this is not a dropped-timezone parser bug. Those convert to 16:00 and 15:00
  Europe/London on October 30 respectively. The option clock conflicts with the expected
  monthly termination alongside the future under the
  [CME standard option rules](https://www.cmegroup.com/content/dam/cmegroup/rulebook/CME/IV/350/350A.pdf).
  No one-hour override was introduced.
- Broker GBP/USD hours showed closure until September 27 at 21:15 UTC / 22:15 London.
  Fresh usable FX must still be received before a GBP budget can be calculated.
- Concrete product/expiry mapping, delta tolerance and the account-specific fee reserve
  remain unapproved. Diagnostic monthly quotes do not approve a substitute expiry,
  strike, Micro Bitcoin or Bitcoin Friday product.

Resolving the remaining mapping requires a verified IBKR contract identity for the frozen
same-day method and confirmation of its termination-time semantics, followed by approval
of the concrete mapping/tolerance/fee configuration. Until then BTC is monitored with
orders blocked. The independent NQ prior-volume history block also remains unchanged.

## Deployment evidence

A fresh read-only preflight at 17:28:32.677419 UTC verified the allowlisted account, unarmed
configuration, no open orders or positions and zero returned executions. The app alone
restarted at 17:28:32 UTC after a consistent SQLite backup. Gateway and the independent
scanner service/timer were unchanged.

[Postflight at 17:31:23 UTC](postflight.json) verified the new release, authenticated routes,
connected/reconciled PAPER state, zero ledger orders/fills/reservations/positions and
13 of 60 app-owned lines. BTC reported current bars and all five reference sessions.
App PID 1441785, zero automatic restarts, approximately 62.4 MiB memory. L2 remained
disabled and LIVE unavailable. No order, what-if order, order cancellation, subscription
purchase or permission change was sent.

Protected backup:
`/var/lib/stocker/backups/futures-cutover-20260927T152213Z/chain-id-fix-20260927T172832Z.sqlite3`.
Diagnostic probes are retained outside the active runtime in that backup directory's
`debug` subdirectory. This is a point-in-time verification, not evidence of future broker
state or a transmitting execution test.
