# September 24 entry failure follow-up

The deployed release remained `b655d4b` throughout this investigation. The source
repairs on `fix/first4-trade-timestamps` are not loaded in that process. No service
restart, manual arming, slot reset, order action or fresh broker-data request was
performed for this follow-up.

## Observed failures (UK time)

| Slot | Symbol | Failure time | Recorded evidence |
| --- | --- | --- | --- |
| 1 | GCDT | 14:48:01 | Anchor recorded, then ambiguous trading-class/multiplier error |
| 2 | MMTIF | 15:02:00 | Timeout at the end of the baseline minute, no anchor recorded |
| 3 | SOS | 15:21:00 | Anchor recorded, then ambiguous trading-class/multiplier error |
| 4 | PMAX | 15:22:00 | Timeout at the end of the baseline minute, no anchor recorded |

The authenticated dashboard showed no orders, positions or exit obligations, and
scanner/manager health remained RUNNING. All four permanent slots were consumed.
The old `broker_trade_time` field actually used receipt time, as demonstrated
against the pinned library; do not treat those historical values as independently
verified broker execution timestamps.

## Confirmed source defects and repairs

- ib_async 2.1 substitutes packet receipt time for the wire trade timestamp.
  The FIRST4 instance adapter preserves the latter before tick callbacks run;
  quote receipt timestamps are unchanged. Offline decoder and execution tests
  reproduce and prevent acceptance of a late previous-minute trade as an anchor.
- Tick subscriptions lacked a request future. Their broker errors could be logged
  without reaching the anchor wait. Registering that wait uses the pinned
  library's existing `RaiseRequestErrors` semantics, preserves request-specific
  errors, and retrieves pending exceptions during cleanup.
- Subscription cancellation left a request-to-ticker mapping. Cleanup now removes
  it along with the callback and future, including failure/cancellation cases.
- Empty, incompatible and ambiguous chain responses shared one error. They now
  have distinct codes and bounded metadata. Checking the cached response before
  the anchor wait avoids spending that minute on a known unusable chain.

The chain filter itself is unchanged: SMART, exact underlying-symbol trading
class, multiplier 100, then all existing qualified-contract/expiry checks. No
adjusted-chain fallback, deadline extension or replacement admission was added.

## What remains unproven

Today's logs did not retain the returned chain rows or per-subscription tick
counts. They cannot prove whether GCDT/SOS had no permitted chain or multiple
matching chains, or whether MMTIF/PMAX had no eligible trade versus missing feed
delivery. The timestamp defect is real, but is not proven to have caused these
four failures. Improved error handling does not manufacture contracts or data.

Loading the patch requires a planned restart. The process has no supported hot
reload. Dated authority and scanner continuity must not be replayed or reset to
activate it. A subsequent explicitly authorized session needs validation of the
actual candidate path using the retained chain/subscription evidence; Ford's
opening quote probe alone does not establish that every stock can execute.

Reference: IBKR documents [tick request parameters and limits](https://www.interactivebrokers.com/docs/tws-api/doc/market-data-live/tick-by-tick-data/request-tick-by-tick-data)
and [request-error codes](https://www.interactivebrokers.com/docs/tws-api/doc/error-handling/error-codes).
The implementation was also checked against the installed, locked ib_async 2.1
decoder, wrapper and request lifecycle rather than changing error flags globally.

## Offline verification

- `rtk .venv/bin/pytest tests/test_first4_anchor.py tests/test_first4_chain_diagnostics.py --tb=short`:
  initially reproduced ten failures, then passed all 18 tests at that stage.
- `rtk bash scripts/check.sh python`: 528 passed, five existing warnings, 59.10s.
  A final change to retain both simultaneous chain/tick errors was then checked by
  the relevant suite below; GitHub CI reruns the full suite on the published commit.
- `rtk .venv/bin/pytest tests/test_first4_anchor.py tests/test_first4_chain_diagnostics.py tests/test_first4.py tests/test_first4_deployment.py tests/test_first4_repairs.py`:
  final patch, 121 passed in 4.04s, including frozen FIRST4 equivalence.
- `rtk bash scripts/check.sh format`, `lint`, `typing`: passed; 190 formatted
  files and 108 type-checked source files.
- `rtk bash scripts/check.sh server`: passed locked server-only installation
  and startup with network connections prohibited.
- Bundled Node ran `tests/dashboard_first4.cjs`: passed security escaping,
  refresh, timeout, ordering and pause-control checks. Local `npm` is unavailable.
- `rtk git diff --check`: passed. Frozen fixtures, config, calculation code and
  lockfiles are unchanged.

Fake-response checks also confirm a standard-chain check reuses the existing
cached response (one request for two checks) and a rejection audit contains at
most 20 row summaries even when 1,000 rows are returned. These are deterministic
application checks, not measurements of production brokerage performance.
