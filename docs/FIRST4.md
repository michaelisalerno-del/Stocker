# Frozen FIRST4 PAPER replacement

Source: the local 2026-09-23-fast-four-trade-chronological-selection-test workspace, not GitHub main. Authoritative `run_test.py`, `protocol.json`, `verification.json`, and the `execution_delay_v0` frozen input, protocol, verification, manifest and delay ledger were inspected. Fixtures preserve 1,293 TOP25 first appearances and the original source functions; all 80 FIRST4 allocations over 20 sessions are compared without running the research experiment. SHA-256 provenance is in `tests/fixtures/first4/source_manifest.json` and the original copied manifests.

The upstream source is `2026-09-23-fast-broad-universe-scanner-depth-test/expanded.py` and `options_replay.py`, using `scanner_filter_v0/historical.py:excursion` for PRIOR15. Native scanner: STK, STK.US, MOST_ACTIVE, changePercAbove=5.5, priceBelow=20, 25 rows. Historical eligibility is change >=5.5 and price strictly <20; broker scanner ranks are retained, never reconstructed from asynchronous replies. The historical volume-rank approximation's stable alphabetical tie order is already encoded in the saved ranks. Production accepts the native rank as requested, rather than promising that a native scanner reproduces a historical approximation of its universe.

PRIOR15 uses the 15 completed minute OHLC bars ending j, with reference close(j-15), or the first session open when j=14. With H=max(high), L=min(low), ref as above, the source performs up=max(0,(H/ref-1)*10000), down=max(0,(1-L/ref)*10000), then (up+down)/100. It is anchor-inclusive range, not absolute return or (H-L)/last close. Invalid/missing bars or reference are unavailable; no interpolation, earlier-window substitution or later readmission. Q5 is strictly >4.459368321659181 percent.

Each stock/session's first native scanner appearance is frozen. Decisions for an entire minute are committed in native rank order after bounded concurrent history reads. The first four Q5 opportunities consume permanent daily slots, even if unarmed, unpriceable, rejected or unfilled. No fifth replacement or capital recycling.

For a bar stamped 09:44 (j=14), the completed information time is 09:45; baseline open(j+2) is 09:46. A live Last-trade stream starts before 09:46; its first eligible trade supplies the baseline strike anchor. Listed contract qualification/quote latency is recorded and never disguised as a historical opening fill. There is no deliberate extra minute or three-minute wait. Restart after missed scanner history blocks further admissions for that session; existing slots and exit obligations persist. Missing baseline anchors consume their slot without later retries.

Research buys put at .98*S0 and call at 1.02*S0, with expiry=baseline entry+2,880 calendar minutes and scheduled session-close valuation. IV=100%, r=.04, q=0 and 1.05/.95 benchmark marks are absent from broker pricing/P&L.

## Required execution settings

The saved research explicitly does not approve listed expiry/strike mapping. The £100 illustration permits fractional packages and is not evidence of an agreed executable premium budget or FX convention. No defaults are inferred. `configs/first4.example.yaml` exposes these required choices:

- `expiry_rule`: EXACT_CALENDAR_DATE or FIRST_ON_OR_AFTER the synthetic expiry date. No fallback if exact expiry is unavailable; ambiguous trading classes/multipliers reject.
- `strike_rule`: OUTWARD or NEAREST_TIES_OUTWARD from .98/1.02 references (the call reference is 1.02*S0).
- `premium_budget_usd` and `fee_reserve_per_package_usd`: actual multiplier and broker quantity increment; floor quantity, never increase to minimum size. No FX rate is invented.
- `entry_limit`: SUM_OF_ASKS, rounded down to broker combo tick; `quote_max_age_seconds` and `entry_deadline_seconds` (less than a minute). Both option quotes must be current real-time bid/ask data. GTD bounds the entry order lifetime. A supported SMART debit combo is used; missing combo execution rules reject visibly.
- `exit_seconds_before_close` and `exit_order: MARKET`: explicit tradable pre-close difference from the synthetic closing mark. Broker leg fills determine proceeds. Unfilled/rejected exits remain visible obligations; an overdue close requires operator intervention, never a fictitious closing fill.

Set all fields before `armed: true`. The account identity is fixed to verified PAPER DUP655399 and checked at every submission and reconnect. Ports or environment labels cannot authorise another account. Unknown broker exposure blocks entries without cancelling unrelated orders. The application never sends account-wide cancellations.

## Persistence and recovery

SQLite stores first appearances, allocations, order reservations/IDs, actual leg executions/fees, positions and close obligations. Reservations commit before socket submission. An ambiguous acknowledgement cannot cause automatic entry resubmission. Reconnect retrieves open/completed orders, executions and broker positions before entry authority returns. BAG status does not manufacture leg fills. Exit quantities belong to their original allocation; partial leg exits have separate remaining obligations. Pause survives restart and is checked at submission.

The scanner and broker/exit manager use separate asynchronous tasks in the same process so history requests do not defer close-out. Contract chains are cached per underlying/session. No universe history download, research calculation, model load, observer dependency or extra candidate scanner is on the path.

Historical runtime databases/configuration are backed up before cutover and kept outside active FIRST4 configuration. The new app does not parse old run files or old runtime state. The independently pinned three-scanner observation service is not changed.

## Verification

Run `pytest tests/test_first4.py tests/test_dashboard_security.py`, the remaining shared/research suite, type checks, `npm test`, and `python scripts/server_smoke.py`. Broker-boundary tests use test doubles only; they are not presented as broker trading. A real unarmed connection/reconciliation check is separately required at deployment. No arbitrary test order is permitted.
