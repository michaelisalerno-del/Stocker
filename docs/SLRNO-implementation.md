# SLRNO implementation and verification

Implementation-phase report, completed before deployment. All screenshots use synthetic
fixtures. The subsequent user-authorized activation of `92393ff` is recorded in
[the current deployment record](CURRENT-DEPLOYMENT.md).

## A. Starting identity

The task directory was empty. Located the current FIRST4 checkout on
`fix/first4-trade-timestamps` at `94c635b538f62c5bfb0df8ab0246a6afe5c4b955` and created
this isolated worktree on `codex/slrno-lean-dashboard`. The recorded server release is
`b5d6962cbd2d13eb39129ed0a116d9f4c87a9cac`; the only difference from the starting commit
is its deployment-record documentation. No server inspection or connection was made.
Today's entry-anchor, opening-retry, diagnostic and option-volume-filter fixes remain intact.

## B. Material findings and implementation

| File / function | Verified starting behaviour | Implemented change |
| --- | --- | --- |
| `dashboard.js` / refresh | Replaced `#main` every three seconds; destroyed scrolling wrappers and controls. Interval already had an in-flight guard. | One shell per navigation; keyed rows, persistent cards/wrappers and changed text only. Completion-based scheduler, abort/timeout, visibility suspension, revision guard and last-good data. |
| `app.py` / overview | Already session-scoped and paginated, but carried raw candidate/order payloads, configuration and several unrelated sections. | Compact Overview; separate Opportunities, Execution, System and on-demand evidence. One allocation projection feeds both cards and economics. |
| `views.py` / allocations, economics | Session accounting already used complete effective fills, but UI fragmented order/fill/position information. | Aggregate complete effective fills by session + symbol + contract, linked through order references. Distinguish partial legs, open exposure, exit pending, zero exposure awaiting reconciliation, closed, unfilled and blocked entries. |
| `views.py` / opportunities | Frozen first appearances, not refreshed PRIOR15 observations. | Six nearest unallocated snapshots, deterministic display ordering, backend Q5, explicit final decisions; 50-row decision pages with a maximum of 200. |
| `dashboard.js` / pauseEntries | Pause lacked deliberate confirmation and reliable distinction from unarmed. | One global confirmation, explicit persisted paused state, disabled duplicates, HTTP checks, retained errors and authoritative reads after ambiguous outcomes. No mutation retries. |
| `index.html`, `dashboard.css`, `security.py` | Stocker branding; obsolete CSS and missing active navigation. | SLRNO identity, four routes, active `aria-current`, visible freshness receipt, readable responsive cards, keyboard/mobile controls, updated authentication realm. Credential identifiers unchanged. |

The launcher starts Uvicorn and FIRST4 on the same asyncio loop, with SQLite owned
there. No threads, caching layer or streaming system were added. Dashboard requests
now avoid full-ledger reads; existing runtime/reconciliation checks remain complete.
Existing session/order/fill indexes suffice for the measured workload; no schema,
index or durability changes were introduced.

## C. Deleted or moved material

- Replaced the fragmented Candidates/Orders/Positions/Trades/Settings UI implementations.
  Old page paths redirect to Opportunities, Execution or System; their old API implementations
  were removed. The current browser uses the new page-specific API only.
- Removed the unused `Store.page()` helper after checking all repository callers. Retained
  `Store.rows()` because reconciliation and the operational PAPER check still use it.
- Removed the old duplicate session-accounting helper; one projection now serves the dashboard.
- Replaced obsolete run-builder, live-zone, run-card, configuration-form and performance-strip CSS.
- Moved scikit-learn **with its existing 1.8.0 pin** and joblib from base to research dependencies.
  Their production-only transitives SciPy and threadpoolctl disappear from that install too.
  No locked versions changed. Research defaults remain `dev` + `research`.
- Corrected stale bootstrap next-step text and the obsolete release hash in README. Historical
  research, frozen fixtures and deployment evidence were not rewritten.

The actual [clean production import inventory](slrno-production-imports.json) includes
NumPy, pandas and exchange calendars, so these remain. MCP/CLI dependencies were retained
for their separate supported entry points. The server already used
`uv sync --locked --no-default-groups --group server`; default research groups were not
being installed by that deployment command.

## D. SLRNO interface

**Overview → Opportunities → Execution → System.** Overview has compact operational
status, actionable warnings, exactly four permanent slots, up to six opportunity snapshots
and session economics. Cards prioritise selection evidence before fills, held legs and
premium after fills, and realised/provisional results after verified closure. Consumed
slots remain occupied through failure, unfilled orders and closure.

Timing/leg details expand in place. Allocation evidence loads explicitly and combines
orders, actual executions/commissions, quoted comparisons and technical evidence.
Evidence pagination does not limit accounting. System includes read-only configuration,
opening verification and bounded, paginated broker-position/obligation diagnostics.

Synthetic screenshots: [desktop Overview](slrno-screenshots/overview-desktop.png),
[mobile Overview](slrno-screenshots/overview-mobile.png),
[Opportunities](slrno-screenshots/opportunities-desktop.png),
[Execution](slrno-screenshots/execution-desktop.png), [System](slrno-screenshots/system-desktop.png).

## E. Data limitations

Candidate proximity is **snapshot-only**, with observation time and permanent decision.
The runtime does not retain successive comparable observations or daily-change values
in these records; neither movement arrows nor daily-change metrics are fabricated.
Q5 equality fails the strict comparison, and a capped bar never implies allocation.

Actual PAPER leg quantities, multipliers, prices, fees and corrected executions support
cash flow and completed allocation results. Missing commissions make closed results and
returns provisional. Return means net result divided by actual premium paid; zero/missing
premium gives no return. No reliable current valuation exists: **unrealised P&L unavailable**.
Quote comparisons remain explicitly separate from broker results. Totals are session-only;
allocation counts and option-leg counts are distinct.

## F. Refresh behaviour

Automatic updates retain wrapper nodes, keyed rows, details, filters, focus, selection
and scroll positions. No unchanged child/text content is rewritten. Detail data is a
labelled snapshot refreshed explicitly. Historical pages load once and subsequently poll
only compact current operational status.

Read timeout: eight seconds. Next refresh begins only after completion: Overview/Execution
five seconds, Opportunities fifteen, System thirty. Hidden pages stop polling; returning
triggers one refresh, including the in-flight-abort case. Failed reads retain data and
show the last successful receipt; recovery clears staleness. Transport receipt is separate
from recorded scanner/quote freshness. Browser visibility never changes trading activity.

## G. Measured performance

Same local ASGI client and synthetic fixture, 31 samples per endpoint: 500 current-session
candidates, four allocations/eight fills; grown fixture adds 100 historical sessions
(50,500 events total). Candidate diagnostics contain 2 KB each. These are local measurements,
not broker-session or internet latency predictions.

| Grown fixture | Response bytes before → after | Median ms | p95 ms | SQL statements | SQLite VM steps |
| --- | ---: | ---: | ---: | ---: | ---: |
| Overview | 370,905 → 7,983 | 2.118 → 0.758 | 2.560 → 1.120 | 11 → 8 | 13,031 → 9,848 |
| Opportunities | 358,899 → 13,119 | 1.917 → 0.729 | 2.331 → 0.922 | 7 → 4 | 11,823 → 7,089 |
| Execution¹ | 11,323 → 6,433 | 0.276 → 0.335 | 0.311 → 0.492 | 7 → 7 | 330 → 1,132 |
| System² | 1,361 → 1,842 | 0.182 → 0.200 | 0.273 → 0.275 | 3 → 5 | 41 → 156 |

¹ Execution now includes complete allocation economics; the old comparison is its orders view.
² System now includes paginated positions/obligations, previously sent to every page.

Overview Python JSON parses: **8 → 4**. Small and grown fixtures have identical response
sizes, statement counts and VM steps in both versions: historical-growth isolation was
already fixed and is preserved. Current-session snapshot ranking still inspects that
session's candidate values. No claim of constant work as the current session itself grows.

Measured Chromium timer tests: old interval **20 requests/minute**; new Overview **12**,
Opportunities **4**, Execution **12**, System **2** with immediate synthetic responses.
Hidden-page requests in one simulated minute: **0**. Maximum active refreshes: **1**.
Horizontal scroll: old **260 → 0 px**, new **260 → 260 px**. New unchanged polls: **0** main
child/text mutations; changed rows retain their nodes, focus and scroll.

Fresh locked dependency installs, excluding project files and bytecode: **59 → 55**
third-party distributions; **198,125,116 → 97,049,377 bytes**. This measures installation
footprint, not steady-state speed. Startup memory/time, real broker callback/deadline latency,
network installation time and literal SQL rows visited were not measured; VM instruction
counts are provided instead. Python parse counts exclude SQLite JSON extraction.

Artifacts: [API before](slrno-benchmark-before.json), [API after](slrno-benchmark-after.json),
[browser before](slrno-browser-before.json), [browser after](slrno-browser-after.json),
[dependency measurements](slrno-dependency-measurement.json).
Reproduce API fixtures with `python scripts/dashboard_benchmark.py --after` in the locked
dev/server environment; run the same script without `--after` with starting-commit source
on PYTHONPATH for the baseline. Browser regression entry point remains `npm test`.

## H. Safety and verification

**Frozen FIRST4 selection/execution semantics changed: NO.** FIRST4 mathematics, scanner,
runtime, broker, configuration and frozen fixtures are unchanged. The only Store change
removes an unused presentation helper. WAL and `synchronous=FULL` are retained and tested.
Pause persistence and execution behaviour remain unchanged; the API additionally reports
its existing authoritative flag. No LIVE capability, new broker request, deployment,
service restart, arming change, or IBKR connection was introduced.

Executed checks:

- Full Python suite: **538 passed**, five existing warnings (one Starlette deprecation,
  four NumPy empty-slice warnings). Relevant FIRST4, security, configuration, execution,
  packaging and research checks included.
- Playwright/Chromium acceptance passed: wide scrolling, vertical position, focus, text
  selection, expanded detail/filter/selection retention, active route, request races,
  timeout/recovery, hidden/in-flight visibility, stale recovery, escaping, mobile menu/Escape,
  pause confirmation/duplicates/ambiguous results, four slots/lifecycle, Q5 edge cases,
  unavailable valuation and mobile overflow. The package's exact Node test entry point was
  run with the existing locked Playwright 1.62.1 installation; npm itself was unavailable locally.
- Ruff lint and format check passed; mypy passed for **109 source files**; `git diff --check` passed.
- Fresh locked production install and offline `scripts/server_smoke.py` passed, explicitly
  checking that scikit-learn/joblib/Jupyter/pytest are absent. Research dependency versions preserved.

## I. Remaining issues

No known failing local acceptance checks. Real broker-session and deployed-browser validation
were not performed, as requested. Snapshot-only candidates and unavailable live valuation are
existing data limitations, documented and represented explicitly rather than inferred away.
