# FIRST4 PAPER execution

Source: the local 2026-09-23-fast-four-trade-chronological-selection-test workspace, not GitHub main. Authoritative `run_test.py`, `protocol.json`, `verification.json`, and the `execution_delay_v0` frozen input, protocol, verification, manifest and delay ledger were inspected. Fixtures preserve 1,293 TOP25 first appearances and the original source functions; all 80 FIRST4 allocations over 20 sessions are compared without running the research experiment. SHA-256 provenance is in `tests/fixtures/first4/source_manifest.json` and the original copied manifests.

The upstream source is `2026-09-23-fast-broad-universe-scanner-depth-test/expanded.py` and `options_replay.py`, using `scanner_filter_v0/historical.py:excursion` for PRIOR15. Original native scanner: STK, STK.US, MOST_ACTIVE, changePercAbove=5.5, priceBelow=20, 25 rows. Historical eligibility is change >=5.5 and price strictly <20; broker scanner ranks are retained, never reconstructed from asynchronous replies. The historical volume-rank approximation's stable alphabetical tie order is already encoded in the saved ranks. Production accepts the native rank as requested, rather than promising that a native scanner reproduces a historical approximation of its universe.

## Approved scanner universe change — September 24, 2026

The user approved adding IBKR's native `averageOptionVolumeAbove=1` field to the
existing scanner request. This is a minimum **average option volume** filter,
not a boolean guarantee of option availability. It is a `ScannerSubscription`
field in the pinned ib_async 2.1 API, not a guessed `avgoptionvolume` tag.
[IBKR field documentation](https://www.interactivebrokers.com/docs/tws-api/protobuf/scanner-subscription).

IBKR applies this restriction before returning its top 25 stocks. FIRST4 records
first appearances and commits slots in that filtered universe's native rank
order. All other scanner fields, PRIOR15/Q5 calculations and execution rules
remain unchanged. A failed filtered scan never falls back to an unfiltered scan.
This adds no per-stock requests or new subscriptions to the application loop.

The original 1,293 appearances and 80 allocations remain unchanged historical
fixtures. Their passing tests validate the original calculations and slot rules;
they do **not** validate performance or selections in the newly filtered universe.
Exact standard chains, approved expiry/strikes and fresh two-sided quotes still
require the existing execution checks. A stock admitted by this filter can still
fail those checks and consume a slot. No fifth replacement is allowed.

Do not reinterpret existing session records using the new universe or clear
today's slots. Activation requires a new process; an interrupted session remains
blocked and a later session needs its existing explicit dated authorization.
See [CURRENT-DEPLOYMENT.md](CURRENT-DEPLOYMENT.md) for activation status.

A separate read-only PAPER diagnostic at 15:24 UTC on September 24 verified that
Gateway accepted this field. Two sequential scans returned 25 original-universe
stocks and 23 filtered stocks; GCDT and PMAX appeared only in the former snapshot.
The first three filtered results (GLND, SNDQ, CRML) each returned one permitted
standard option chain. This is a current API observation, not historical strategy
validation or proof of executable strikes/quotes. No order actions were attempted;
the diagnostic client disconnected with no outstanding request/subscription state.
The application and independent observer continued without restart.

## Retained calculation and allocation rules

PRIOR15 uses the 15 completed minute OHLC bars ending j, with reference close(j-15), or the first session open when j=14. With H=max(high), L=min(low), ref as above, the source performs up=max(0,(H/ref-1)*10000), down=max(0,(1-L/ref)*10000), then (up+down)/100. It is anchor-inclusive range, not absolute return or (H-L)/last close. Invalid/missing bars or reference are unavailable; no interpolation, earlier-window substitution or later readmission. Q5 is strictly >4.459368321659181 percent.

Each stock/session's first native scanner appearance is frozen. Decisions for an entire minute are committed in native rank order after bounded concurrent history reads. The first four Q5 opportunities consume permanent daily slots, even if unarmed, unpriceable, rejected or unfilled. No fifth replacement or capital recycling.

For a bar stamped 09:44 (j=14), the completed information time is 09:45; baseline open(j+2) is 09:46. A live Last-trade stream starts before 09:46; its first eligible trade supplies the baseline strike anchor. Listed contract qualification/quote latency is recorded and never disguised as a historical opening fill. There is no deliberate extra minute or three-minute wait. Restart after missed scanner history blocks further admissions for that session; existing slots and exit obligations persist. Missing baseline anchors consume their slot without later retries.

The FIRST4 broker adapter preserves the trade timestamp supplied on the wire;
the pinned ib_async 2.1 wrapper otherwise substitutes packet receipt time.
Quote receipt timestamps remain unchanged. A late-arriving trade from before the
baseline minute cannot supply the anchor. Execution failures record their stage,
exception, observed tick count and last trade timestamp. A baseline timeout is
reported as `BASELINE_TRADE_NOT_RECEIVED`; this does not assert whether the market
had no trade or the feed failed to deliver it. No later-price fallback is used.

The already-requested option-chain metadata is checked before waiting for the
baseline anchor. An empty broker response (`OPTION_CHAIN_EMPTY`), no matching
standard chain (`NO_PERMITTED_STANDARD_OPTION_CHAIN`), and multiple matching
chains (`AMBIGUOUS_STANDARD_OPTION_CHAIN`) are distinct failures. Rejection
details include the required mapping and at most 20 returned chain summaries,
without full strike/expiry lists. The same exact chain restrictions apply again
during contract selection, reusing the cached response. This changes neither
first-appearance admission nor the permanent slot consumed by a failed execution.

The anchor wait participates in ib_async's request-error lifecycle, so a rejected
tick subscription records `BASELINE_SUBSCRIPTION_FAILED` with the broker request
ID, code and message. Informational notices and other request IDs do not fail
that wait. Cleanup removes its callback, request future and subscription mapping
on completion, failure or cancellation. No subscription retry or later anchor is
introduced.

Research buys put at .98*S0 and call at 1.02*S0, with expiry=baseline entry+2,880 calendar minutes and scheduled session-close valuation. IV=100%, r=.04, q=0 and 1.05/.95 benchmark marks are absent from broker pricing/P&L.

## User-specified PAPER execution conventions

These conventions were specified by the user after the initial deployment. They are **not validated by the synthetic research**. The historical £100 fractional sizing example is not used. Exact configuration is in `configs/first4.example.yaml`; `armed` remains false until the required non-transmitting checks pass.

| Configuration | Value / units |
|---|---|
| `expiry_rule` | `NEAREST_WITHIN_24H_LATER_TIE` |
| `strike_rule` | `NEAREST_STRICT_OTM_WITHIN_1PCT` |
| `packages_per_candidate` | 1 put + 1 call |
| `premium_budget_usd` | 250 USD maximum combined premium |
| `fee_reserve_per_package_usd` | 10 USD separate round-trip allowance |
| `session_allocation_usd` | 1,040 USD, including pending orders and reserves |
| `entry_limit` | `SUM_OF_ASKS`, rounded down to the broker combo increment |
| `quote_max_age_seconds` | 5 seconds for each bid/ask price observation |
| `entry_deadline_seconds` | 180 seconds after the original baseline entry |
| `exit_seconds_before_close` | 120 seconds before scheduled stock-session close |
| `exit_order` | `MARKET`, the existing close-out route, after executable quote checks |

`execution_delay_v0/run_delay.py` fixes expiry to baseline + 2,880 calendar minutes; `options_replay.py` and the original model confirm `YEAR=365*1440`. Actual listed expiry timestamps must be within ±1,440 calendar minutes of that target, with the nearest selected and exact ties preferring later expiry. Same-day New York expiries are excluded. The selected put and call share the date and actual broker expiry time. The broker's `realExpirationDate`, `lastTradeTime` and `timeZoneId` supply the actual timestamp and remaining seconds. At most three specific expiry contracts are checked; no expiry-time substitute or wider search is used.

Strikes retain the baseline first trade in the original open(j+2) minute as their reference, never a later quote. Each nearest strictly OTM listed strike must be within .01 times that reference of its .98/1.02 target; exact ties prefer further OTM. The contract must have the same underlying, USD currency, multiplier 100, standard underlying trading class, and exact unadjusted OSI symbol; inconsistent/adjusted metadata rejects. Actual expiry, strikes and mapping differences are recorded.

Exactly one pair is attempted. A durable full $260 allocation is reserved before socket submission and never recycled that session, including cancellation or partial fill. Broker quantity increments must permit one. A fifth admission cannot be manufactured by an execution failure. Actual commissions are recorded separately from the $10 allowance; unknown commissions are not represented as zero final fees.

Both quotes must be real-time (type 1), strictly positive/non-crossed, with sufficient ask size for entry. Freshness comes from bid/ask price observations, not ticker heartbeat timestamps. A SMART debit-combination GTD order is submitted as soon as preparation permits at/after the original scheduled time, with no deliberate delay. The new 180-second deadline is an execution window, not a three-minute strategy delay. At the deadline, the manager explicitly cancels only its remainder and waits for cancellation/fill reconciliation; no separate-leg entry fallback exists.

The existing market close-out begins 120 seconds before the calendar's scheduled close, including shortened sessions. Current two-sided quotes with sufficient bid size are captured for the actual remaining legs before submission. A balanced position closes through a combo; an unmatched partial entry has explicit individual leg exit obligations. Account checks always apply, but entry budgets, arming and entry-time restrictions do not block closing. A failed, unfilled or overdue exit is an explicit operator exception; the manager retains the obligation and never fabricates closure or submits after the stock-session close. There is no automatic overnight strategy or hidden retry of an ambiguously acknowledged order.

The dashboard separates **IBKR PAPER simulated fills/P&L and actual fees** from **quoted ask-to-bid comparisons**, whose bid/ask timestamps are retained. Neither synthetic 1.05 entry nor .95 exit multipliers are applied. This two-minute pre-close execution convention differs from the research's scheduled closing valuation.

The account identity is fixed to PAPER DUP655399 and API execution client 81, checked at every submission and reconnect. No LIVE account or alternate execution client can be configured. Unknown exposure blocks entries without cancelling unrelated orders. `scripts/first4_paper_check.py --config /etc/stocker/v1/first4.yaml` uses a separate read-only diagnostic connection with both order methods disabled. It does not allocate a FIRST4 slot, arm the service, or place a test trade.

## Persistence and recovery

For the explicitly requested opening cutover, set `armed: false` and
`arm_after_quote_check_on: 2026-09-24` in the existing configuration. This permits
automatic activation only for that dated US session; it is not a recurring arm
switch. The same service continues to record every scanner first appearance
while a separate asynchronous check verifies the exact PAPER account, fresh
reconciliation with no exposure/open orders, qualified standard Ford diagnostic
options, BAG tick size and fresh real-time two-sided option quotes. Ford's ATM
contracts are only data-access probes and never enter the candidate ledger.
No orders, including what-if orders, are sent by the check.

Qualification has a 30-second bound and BAG metadata an 8-second bound. The
dated check permits at most three sequential attempts for timeouts or temporarily
unavailable stock/option quotes or BAG pricing, with five seconds between attempts.
The first two attempts are capped at 60 seconds including reconciliation. The
final attempt retains the original quote-wait window, ending at stock open + 14
minutes (14:44 UK on September 24), ahead of the first possible frozen Q5 admission
at open + 15 minutes and baseline entry at open + 16. All attempts must finish
before that deadline. Cancellation cleans up each attempt before another begins.
Identity/ownership errors, ambiguous contracts, missing scanner observations,
pause, disconnect/data interruption and revoked permission never permit retry
or restored authority. There is one dated check, no new scanner, no repeated
successful chain downloads, and no scheduled agent needed. A pass
enables the existing broker entry path in memory without restarting; the YAML
retains `armed: false` and the dashboard distinguishes configured and effective
arming. Every actual candidate still undergoes its own contract, quote, budget
and account checks. The check cannot replay or replace an admission.

The dated check and result persist in `opening_check:<date>`, including attempt
count, last temporary error and next attempt time. Status remains CHECKING during
backoff so restart cannot replay it. Exhausted/nonretryable failure, pause,
missed scanner minute or deadline expiry leaves entries disabled. A disconnect
revokes the in-memory authorization. A process restart never restores authority
from the audit record and never repeats an interrupted or completed check.
The authorization expires on date change. Closing obligations continue
independently of arming and readiness. Remove/change the dated setting explicitly
to authorize another session; do not set `armed: true` to bypass a failed check.

SQLite stores first appearances, allocations, order reservations/IDs, actual leg executions/fees, positions and close obligations. Reservations commit before socket submission. An ambiguous acknowledgement cannot cause automatic entry resubmission. Reconnect retrieves open/completed orders, executions and broker positions before entry authority returns. BAG status does not manufacture leg fills. Exit quantities belong to their original allocation; partial leg exits have separate remaining obligations. Pause survives restart and is checked at submission.

The scanner and broker/exit manager use separate asynchronous tasks in the same process so history requests do not defer close-out. Contract chains are cached per underlying/session. No universe history download, research calculation, model load, observer dependency or extra candidate scanner is on the path.

Historical runtime databases/configuration are backed up before cutover and kept outside active FIRST4 configuration. The new app does not parse old run files or old runtime state. The independently pinned three-scanner observation service is not changed.

## Verification

The source repairs described in [FIRST4 repair verification](FIRST4-repair-verification.md)
add supervised worker/manager health, conservative reconnect handling, bounded
ledger/dashboard reads and cancellation cleanup. They are **not deployment
evidence**. `/api/health` retains the dashboard's existing authentication and
returns 503 when execution tasks or persistence are unhealthy; `/api/system`
can still report in-memory failure state when normal ledger views fail.
Task liveness is independent of exchange opening hours. Effective arming means
permission to prepare an entry; every submission still requires its own fresh
quotes, account/ownership checks and frozen execution window.

Dashboard results default to the runtime session (or the latest recorded
session before startup). A session date selector and `session`, `limit` (at most
200), and `offset` query parameters retrieve historical pages. Outstanding
obligations remain visible across sessions. Completed allocations with known
fees contribute realised results even when another allocation remains open;
completed gross results, partial-close gross results, pending fees and cash
flow are separate fields.

The same ledger receives additive indexes and completion/correction markers.
Completion requires reconciled entry deadlines, terminal orders, complete
reported leg executions where required, and matching zero owned/broker
quantities. Late executions/corrections reactivate management; late fees update
accounting without creating another exit. Rejected or ambiguously acknowledged
exits remain operator exceptions under the existing exit policy. No new retry
policy or overnight strategy is introduced.

Run `pytest tests/test_first4.py tests/test_dashboard_security.py`, the remaining shared/research suite, type checks, `npm test`, and `python scripts/server_smoke.py`. Broker-boundary tests use test doubles only; they are not presented as broker trading. A real unarmed connection/reconciliation check is separately required at deployment. No arbitrary test order is permitted.
