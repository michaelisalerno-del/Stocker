# SLRNO review fixes — 29 September 2026

Branch `codex/slrno-review-fixes`, based on `codex/slrno-cleanup` at `a555712`. Nothing was deployed,
no service was restarted, execution was not enabled and no broker request was made. Each stage is
one commit with its own regression tests.

## Changes

1. **Dependencies.** `fastapi`, `starlette`, `uvicorn` and `websockets` are base dependencies: the
   runtime and dashboard import them, so a plain `uv sync` no longer leaves 94 runtime tests
   uncollectable. `mcp` has its own group, `pandas-market-calendars` moved to `research`, and the
   unused `prometheus-client` is gone. The runtime-only install is now
   `uv sync --locked --no-default-groups` (**the `server` group no longer exists**) and contains 28
   packages instead of 54, with no pandas, numpy or mcp. The server smoke asserts that and imports
   the real CLI entry point.
2. **Execution state.** A SQLite error during a dashboard read returns 503 without setting a
   permanent broker fatal error; the runtime's own ledger access still fails closed. A `KeyError`
   during option selection or entry is recorded as `UNEXPECTED_MISSING_FIELD` rather than as a
   coded trading reason. GC is labelled `MONITOR_ONLY` only while blocked, on both pages. GC stays
   monitor-only until its execution mapping is approved, exactly as the rulebook states; a proposed
   config-level GC rejection was **not** made because it would contradict that approval route.
3. **Broker cadence and logging.** Idle broker state is reconciled every 30 seconds instead of every
   two-second cycle; pending orders, held exposure or an unreconciled state still reconcile every
   cycle. `SAXO_SIM` entries require a reconcile within the last 60 seconds
   (`RECONCILIATION_STALE`). Internal paper cycles do not rewrite unchanged positions. A repeated
   management exception is audited and logged once per transition. `slrno.log` has timestamps and
   levels, logs the application's own coded reasons (other exception text stays suppressed) and
   also reaches the service journal. A failed reference-session load retries after 15 minutes
   instead of refetching every session every minute; the reference summary is linear time with
   identical results.
4. **Dashboard.** A paused, otherwise-ready market shows `ENTRIES_PAUSED` on Markets. History rows
   carry summary columns only (evidence stays behind `/api/detail`); Markets reads eight recent
   signals plus one context. Page assets use `Cache-Control: no-cache`. The arm endpoint validates its
   body (422, not 500). The browser fixture now emits production shapes, so two fixture-only
   fallbacks in `dashboard.js` were removed.
5. **Operations.** `scripts/ledger_backup.py` makes a WAL-consistent ledger + configuration bundle
   with the SQLite backup API, integrity/foreign-key checks, provenance, row counts and hashes, and
   can re-verify one. The IBKR-era `backup_state.py` is archived with the parked runtime.
   `saxo_preflight.py` takes the ownership lock before opening the ledger.
6. **Simplification.** Fresh ledgers are created with the current reservation schema; the legacy
   migration is unchanged. Removed: IBKR-style dotted execution revisions (Saxo corrections arrive
   as cumulative order evidence), unused `recover_depth()`, an index on a JSON key the recorder never
   writes, `/api/status`, the `/trades` alias, duplicate status keys and dead front-end code.
   `cost_estimate()` and `budget()` share one penny calculation. Repeated literals are named
   constants; the pinned config fields are documented as sentinels. The CLI no longer imports
   research settings at start-up.
7. **Tests and CI.** Shared fixtures live in `tests/saxo_support.py`; no test imports another test.
   The browser test writes screenshots to a temporary directory unless `--screenshots <dir>` is
   given. CI format/lint jobs install only the dev group, uv caching and a 20-minute job timeout are
   enabled, the pre-commit ruff hook matches the locked ruff, and bare `mypy` works.

## Rollout notes (for a separately authorised deployment)

- Install with `uv sync --locked --no-default-groups`; `--group server` now fails.
- Opening an existing ledger drops the unused `depth_capture_status` index and otherwise changes
  nothing. The previous release recreates the index if rolled back; no data migration is involved.
- Runtime errors now also appear in the service journal. Expect far fewer portfolio REST reads
  while idle; with a pending order or open position the cadence is unchanged.
- Back up with `scripts/ledger_backup.py --database <ledger> --config <yaml> --output <new dir>`.

## Checks actually executed here

- Full Python suite: **518 passed**, 7 existing third-party/numeric warnings. New regression tests
  were run against the pre-change code and failed there, apart from behaviour-preservation checks
  guarding refactors (Markets recent signals, reference-summary equivalence).
- Ruff format (208 files) and lint passed; mypy passed on 117 source files (and bare `mypy`, 116).
- All three Playwright scripts passed with Node 24 and Playwright 1.62.1 (run directly; no `npm` on
  this host's PATH).
- Runtime-only locked install smoke passed in a fresh environment with network connections refused.
- Recorder benchmark, 5 futures + 5 options over two virtual hours: **24,755 archived rows with the
  same SHA-256 as the cleanup's after-run**; ingest p99 2.11 → 2.00 ms.
- Dashboard benchmark (five clients × 40 refreshes, 400 closed trades, 0 broker requests):
  Opportunities response 62,314 → 22,223 bytes; other pages ~107 bytes smaller; workload CPU
  0.130 → 0.112 s. `import stocker_core.cli`: ~174 ms → ~50 ms.

These are offline measurements on a development machine, not production evidence.

## Not changed, deliberately

- Per-market awaits inside `decisions()` stay sequential; revisit before any mapping is approved.
- The recorder still packs a futures record twice when adding book-flow features; changing it
  risks the archived evidence format for a sub-millisecond gain.
- `option_view` still calls `cost_estimate` twice (limit and ask) rather than duplicating arithmetic.
- CSS breakpoints were not consolidated (visual change at specific widths); the websocket branch of
  the security middleware stays (it is tested and covers any future websocket route).
- Outside the web app: the research CLI (~3.5K lines) inside `stocker_core`, the unused server
  config chain, eight unused research dependencies and the retired EODHD transport tests remain.
