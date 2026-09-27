# Market-data/L2 implementation validation — 2026-09-27

Local branch `codex/futures-paper-replacement`, extending futures commit
`4eff5fc3b55f703bd78cf1794a25561847413f9b`. Implemented, not deployed; no broker connection,
orders, account purchases, arming, remote configuration changes or service restarts for this work.

Changed components: `subscriptions.py` owns feeds/quotas/history, `pacing.py` owns the one wire
scheduler, `depth.py` reconstructs/allocates/records books, `requests.py` handles installed API
callbacks and bounded requests. Broker/runtime/config/store integration preserves execution
ownership and adds coverage metadata. Dashboard cards, System/API panel, preflight output,
offline fixtures, README and current deployment documentation are updated. No new dependency,
service, broker connection or strategy variant was added. Frozen `rules.py` and `contracts.py`
are unchanged; existing account, £10/one-contract/four-trade/£40 safeguards remain.

| Validation | Result |
|---|---|
| Full Python suite | **441 passed**, 66.05 seconds; five existing dependency/research warnings |
| Final focused futures/data suite after final recovery/label changes | **52 passed** |
| Ruff formatting / lint | **188 files formatted; lint passed** |
| Mypy | **111 source files passed** |
| Browser | Six fixed cards, depth states/ladder freshness, focus, expanded details, filters, vertical/horizontal scroll, mobile and System passed |
| Server-only smoke | Fresh temporary locked installation, imports, CLI, authenticated dashboard startup/assets passed |
| Active runtime/startup text search | No retired strategy imports, names or consumed-slot dependencies in packages/apps/scripts/configs/.github/README/current deployment instructions |
| Source parity | Existing frozen signal/exit, GC policy, expiry, calendar/DST, integer budget and reconciliation fixtures passed |

Resource fixture: six futures L1 + six streamed bars + FX = **13**; adding three books = **16**.
Four owned options, four retained underlyings, fifteen simultaneous candidate quotes and three books
total **39**. One serial pre-admission quote handoff produces **40**; the configured app ceiling is
60. The fixture releases all owners and confirms no retained request/ticker/bar registrations.
Reduced external capacity blocks optional candidates while preserving existing exposure data.

Pacing fixture: after 30 core messages, ten urgent cancellations transmit ahead of 64 queued
optional messages; total remains 40 in the rolling second. Expired optional requests are removed,
never transmitted later. Depth-only error 309 does not block a valid option-selection quote.
There are no transmitted broker test orders: all wire/order interfaces in these tests are fakes.

Recording-pressure fixture: **100,000 callbacks in 0.231 seconds**, including cheap early returns
after optional recording reaches its cap. This is an overload-shedding measurement, not a claim of
sustained complete-book recording at that rate. At a 262,144-byte configured accounting ceiling,
peak accounted memory was **212,992 bytes**; traced new Python allocations peaked at **15,668 bytes**.
Recording paused with `RECORDING_MEMORY_LIMIT`, retained its explicit gap, and performed **zero SQL
statements inside the callback loop**. Disk-limit rejection and retained opportunity denominators
also passed. The standard configured memory/disk ceilings remain 32 MiB / 256 MiB.

L2 isolation: disabled, enabled and permission-failed modes produce identical frozen event/order
results: same PAPER account, long-option side, quantity, limit and £10 reservation. L2 has no
argument, import or awaited dependency in the rule/order-submission path. GC has no experimental
management; opportunity exit anchors and admission order remain unchanged. Received coverage and
complete-book coverage are separate; post-trigger rows never become pre-trigger context.

## Standards review

Two confirmed findings were fixed: unanswered history retaining slots, and an old cancellation
removing a restored subscription with a reused request ID. The reviewer confirmed bounded request
cleanup, generation ownership and all three focused recovery regressions. No remaining finding
within those fixes.

## Requirements review

Findings around existing-feed priority, cancellation propagation, core/depth ownership after
preemption, depth-only capacity backoff and immediate gap marking were fixed. Eight targeted
regressions passed on independent review; no remaining issue within the reported fixes.

Actual screenshots, all labelled **OFFLINE TEST FIXTURE**:

- [Six-market overview](futures-screenshots/overview-desktop-fixture.png)
- [Expanded observation-only ladder](futures-screenshots/depth-expanded-fixture.png)
- [System/API allowances and separate depth slots](futures-screenshots/system-api-fixture.png)
- [Mobile overview](futures-screenshots/overview-mobile-fixture.png)

Actual broker allowance and subscriptions remain unverified in this task. Baseline 100 lines,
three depth slots, five disabled tick-by-tick slots and 50 requests/second are documented planning
assumptions; external usage is unknown. Non-transmitting deployed preflight must verify entitlement,
real FUT/FOP routing and received bar/depth semantics against the allowlisted PAPER account.
All six markets still require their original approved real-product/expiry/tolerance mappings.
The app remains unarmed with L2 disabled; deployment and eventual arming require the separately
reviewed cutover procedure. Earlier flat/account reports are not proof of present exposure.

See [configuration, exact observation policy and official IBKR sources](MARKET-DATA.md).
