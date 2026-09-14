# Market readiness investigation — 14 September 2026

Later update: [a fresh PAPER Gateway login resolved LSE scanner warning 492](lse-session-refresh-20260914.md). All five scanner families passed on recheck. The historical investigation below is retained.

Follow-up: the user authorized the separate availability policy after this investigation.
[V10 is implemented and prepared locally](candidate-availability-fix-20260914.md); the original
investigation below records the earlier state. Neither the V10 release nor its replacement
run configurations has been deployed.

## Result and release boundary

The required-risk-field form fix is deployed as `c8a3753`. It opens the collapsed risk
section and focuses invalid required inputs before any start request. The Korea run was
successfully created after the user supplied the required gross-notional ceiling.

The working branch `fix/market-readiness-20260914` contains the earlier FX correction
(`e8c39d2`) and the permanent pre-open pause recovery fix. These backend changes are not
deployed. This investigation has not changed subscriptions, routing, candidate rules,
account settings, orders or the saved failed opening sessions.

## Bugs corrected locally

### Transient FX failure was frozen too early

Acquisition preparation could request the cap-conversion FX quote hours before the market
opened and persist its failure into every later capped scanner request. Non-USD acquisitions
now obtain FX near each sweep, with bounded retries inside the existing timing window.
Successful later quotes can resolve still-pending filters; completed/failed observations
remain immutable. Uncapped requests remain independent of FX, and genuinely unavailable FX
still fails the affected components. This does not provide ASX stock-data entitlement.

### Pausing an untouched pre-open session permanently degraded it

The form-only deployment exposed this on the US run: stopping acquisition while all 105
components were still pending sealed the acquisition as interrupted, and the candidate
pipeline persisted a terminal interrupted state. Re-enabling the run could not resume it.

The new regression first reproduced the sealed session and DEGRADED candidate state.
The fix preserves pending acquisition only before the open and only with no observations.
The provider drains cancellation and proves acquisition is untouched; the pipeline independently
checks candidate state and timing, then appends an audit record without resetting evidence.
Tests exercise disable/enable, stop/start, a new pipeline after restart, non-USD acquisition,
and cancellation during the initial SQLite transaction. At-open cancellation and missed
windows still fail closed. Existing cancellation tests protect frozen pools from late writes.

The one-time operational US recovery performed before this work is backed up and audited on
the server. No failed LSE/ASX/Korea selection was repaired with later data.

### Late scanner database writes could change sealed evidence

The full suite exposed a cancellation race in the existing frozen-pool test: a RUNNING
component write completed after the acquisition was sealed. Two deterministic regressions
then held an actual scanner write in a worker thread until after cancellation and sealing;
both RUNNING and DATA_RECEIVED writes changed frozen components/raw hits before the fix.

Scanner callers are now cancelled and drained before their qualification workers. Because
cancelling an asyncio caller cannot stop an already-running thread, component updates and
raw-hit insertion also share an atomic SQLite seal guard. New component plans cannot be
inserted after sealing, and pool insertion checks its seal within a write transaction.
Already-saved plans remain readable; late writes cannot alter the frozen observations.

## LSE — entitlement improved, coverage and scanner warning remain

Original opening observation, 07:00–07:04 UTC (08:00–08:04 London):

- 105 completed scanner components, each with IBKR scanner precision warning 492.
- 458 qualified acquired stocks: 202 complete prefixes, 38 partial, 218 with no opening bars.
- The saved stage was DEGRADED. Its 250 ranked slots included 48 missing-score identities;
  those flags did not represent valid candidates or permission to trade.

After the subscription change and competing-session logout, the retrospective recheck of
all **458 original identities** found **292 complete, 8 partial and 158 with no opening bars**.
All original 202 complete prefixes remained complete; another 90 became available.
[The full per-stock CSV](lse-opening-coverage-20260914.csv) preserves original observed bar
counts, recheck timestamps and current missing minutes/errors. Original partial counts come
from the recorded history diagnostics; failed candidate-stage input arrays alone are empty
and would incorrectly label every original partial prefix as zero bars.

This is later availability, not proof of availability before the original selection deadline.
The diagnostic used the saved conIds and SMART routing, 1-minute TRADES/RTH history ending
at 07:05 UTC, with four concurrent requests. It wrote no runtime/history-cache state.

Additional direct-versus-SMART checks at approximately 09:01–09:02 UTC:

| Stock | Quote result on both routes | Exact opening history result |
| --- | --- | --- |
| FTC (29622960) | Real-time type 1 | No data on both routes; later trades available from 07:50 |
| GCM (37099265) | Real-time type 1 | No data on both routes; longer request contained only prior-session bars |
| JLP (70100847) | Real-time type 1 | No data on both routes in the tested opening and longer interval |
| CAM (35150728) | Real-time type 1 | Both routes start at 07:02; 07:00 and 07:01 are absent |

The empty responses above contain IBKR error 162 explicitly reporting HMDS no data, not
a permission-denied message. These observations support sparse/late opening trade coverage
as a cause for the tested stocks, but do not establish every missing stock's cause or prove
that no trades occurred anywhere. A subscription cannot be assumed to create absent bars.
IBKR documents that its historical trade feed filters some trade types; the application must
use the returned data and cannot manufacture bars. [IBKR historical bar documentation](https://interactivebrokers.github.io/tws-api/historical_bars.html).

Scanner isolation tested all five production families with both `stockTypeFilter=CORP` and
an empty stock-type filter: 10 requests, each returning 50 rows **and warning 492** naming
United Kingdom (LSE) real-time data. The CORP filter is therefore not the cause. Successful
individual quotes/history do not establish precise scanner entitlement. A new diagnostic
API client does not refresh the Gateway's authenticated session. Cached entitlement versus
additional scanner scope remains unresolved; there is no evidence here that buying IOB or
Level II is the required remedy.

[Scanner check timestamps, row counts and exact warnings](market-scanner-permissions-20260914.csv)
cover all 20 LSE/Korea diagnostic requests.

## Korea — access works, complete opening coverage is not universal

All five production scanner families were tested with and without the CORP filter.
HOT_BY_VOLUME, TOP_OPEN_PERC_GAIN, TOP_OPEN_PERC_LOSE and TOP_VOLUME_RATE returned 50 rows
each without precision/permission warnings. TOP_TRADE_RATE returned an empty result on both
tests after the regular close; this is not proof of its opening-session behavior.

The union of the four nonempty CORP scans contained 162 identities. Retrospective exact
00:00–00:04 UTC opening history was **complete for 149, partial for 11 and absent for 2**.
[The full sample CSV](korea-opening-coverage-20260914.csv) lists identities and missing minutes.
This was an after-close diagnostic sample, not the saved opening acquisition, a full cap/sweep
matrix, or an estimate of next session's exact candidate population. The earlier Samsung and
SK Hynix checks also returned real-time quotes, five opening bars and prior daily history;
KRW FX was available. No order or order preview tested trading permissions.

The actual Korea run was created after its first selection window. Its current
`CANDIDATE_SELECTION_WINDOW_MISSED` is expected and must not be retrospectively cleared.
It remains enabled for the next eligible session, but the coverage findings below still apply.

## Remaining selection-policy decision

V9 explicitly treats any acquired stock's incomplete exact prefix as a failed PAPER stage.
This is documented frozen behavior, not a missing capacity setting. Therefore:

1. The same type of failure can recur on another day in LSE or Korea despite working data access.
2. Increasing the initial pool alone cannot solve it: one incomplete member can still block
   the stage, and a larger population creates more history work.
3. Later history cannot restore today's causal selection.

The concrete alternative to evaluate is a **new versioned PAPER policy**: reject and audit
stock-local absent/incomplete prefixes at each stage, rank only exact valid prefixes using
the existing Range5/RV10/RV15 formulas and deterministic ties, and never reintroduce rejected
stocks or replenish the final watchlist after strategy rejection. Global connection,
permission, pacing and deadline failures must remain distinct from local absent data.
Retain the original V9 runs and hashes, compare the alternative against the delayed oracle,
and label coverage/selection differences explicitly. This policy is proposed, not implemented
or silently applied to saved runs. The current correction does not claim to solve this policy
constraint.

## Broker follow-up draft — not sent

> PAPER Gateway API version 178 is using market-data sharing after UK LSE Equities NP L1
> subscription activation and logout of the competing live session. Direct LSE and SMART
> quotes return marketDataType 1, and direct LSE opening history works for tested stocks.
> Every STK.EU.LSE / STOCK.EU scanner family still returns warning 492 naming United Kingdom
> (LSE) real-time market data. Tested HOT_BY_VOLUME, TOP_OPEN_PERC_GAIN,
> TOP_OPEN_PERC_LOSE, TOP_TRADE_RATE and TOP_VOLUME_RATE, with CORP and no stock-type filter.
> Please identify the exact missing scanner entitlement, or confirm whether a Gateway logout
> and login is required to refresh scanner permissions. Does this location include products
> outside the UK LSE Equities L1 subscription? We have not assumed IOB/Level II is necessary.

## Final verification

The final full Python suite passed: **1,224 passed, 14 skipped, 10 warnings** in 110.23 seconds.
The warnings are existing dependency/calendar/research warnings. Ruff lint and formatting
(280 files), mypy (146 source files), all three dashboard browser suites and the fresh
server-only installation/offline startup smoke test passed. `git diff --check` passed.
No temporary debug logging remains in the changed source.

Reproduction commands (run before their fixes and subsequently verified green):

```sh
rtk .venv/bin/python -m pytest tests/test_scanner_acquisition.py -k 'preopen_pause_resumes or pause_never_reconstructs'
rtk .venv/bin/python -m pytest tests/test_scanner_acquisition.py -k pause_during_initial
rtk .venv/bin/python -m pytest tests/test_scanner_acquisition.py -k late_scanner_database_write
```

The investigation's read-only server verification confirms deployed revision c8a3753,
connected/reconciled/ready PAPER,
zero positions and zero open orders. US is READY/enabled; LSE is DEGRADED/enabled from its
original opening; Korea is DEGRADED/enabled from its late start; ASX remains disabled.
Gateway was not restarted for these probes.
