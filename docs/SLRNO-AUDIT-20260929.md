# SLRNO audit fixes — 29 September 2026

Branch `codex/audit-fixes` from `98a772c`, seven commits, each with its own regression tests.
Nothing was deployed, no service was restarted, execution was not enabled and no broker request
was made. The audit reviewed the whole runtime against [AGENTS.md](../AGENTS.md); every finding
below was reproduced in code before it was fixed.

## Trading-critical fixes (`2d91f72`)

| Finding | Fix | Regression test |
|---|---|---|
| `decisions()` took the time once per pass and reused it across per-market awaits; a quote refreshed meanwhile had a negative age and read as `QUOTE_STALE_OR_UNAVAILABLE` | The time is taken per clock, before the gates | `test_quote_refreshed_during_an_earlier_market_await_is_not_stale` |
| Float tick arithmetic (`floor(bid/tick)*tick`) put paper fills two ticks off in 134/400 GC grid prices and produced off-grid `OrderPrice` values | `contracts.grid_price` snaps in Decimal; entry limit and exit price use it | `test_entry_limit_and_exit_price_stay_exactly_one_tick_from_the_quote` |
| One transport error during option-metadata refresh marked the owned contract untradable for fifteen minutes, blocking its exit into `FAILED_CLOSURE_REQUIRES_OPERATOR` | A `SaxoError` keeps the verified entry and does not advance `received_at`; schema failures still block | `test_transport_failure_during_metadata_refresh_keeps_the_verified_entry` |
| OAuth raised bare `ValueError`, which the stream cleanup's `suppress(SaxoError)` let escape: one failed refresh stopped the runtime | OAuth raises the coded `SaxoError` (now defined in `saxo_auth`) | `test_stream_loop_survives_a_lost_token` |
| A 401 only flipped `oauth.status`, which the next `access_token()` reset a second later | A 401 expires the token so the next call refreshes; a failed refresh stays expired | `test_rejected_access_token_forces_a_refresh_and_a_failed_refresh_stays_expired` |
| A dropped connection during refresh discarded the refresh token (it cannot have rotated) | Transport errors keep the tokens; server rejections still clear them | `test_transport_failure_during_refresh_keeps_the_refresh_token` |
| Dashboard `preflight()` reconciled without the management lock | `preflight()` takes `broker.lock` like `manage()` | `test_preflight_waits_for_the_management_lock` |

## Stream and broker state (`21d9077`)

- Token renewal re-authorises the open streaming context (`PUT …/streaming/ws/authorize`, 202
  Accepted) instead of tearing every subscription down for 30–50 s every ~18.5 minutes.
  **Verified on SIM** after deployment: the 18:25:49 UTC renewal kept `reconnects` at 1
  ([postflight](saxo-audit-fixes-deployment-20260929.json)).
- A temporarily disabled subscription, a `_resetsubscriptions` naming targets, or a single silent
  subscription now affects that subscription alone; a quiet session or several silent feeds still
  reconnect. `disabled_targets` clear with the context. FX discovery problems are named
  (`GBPUSD_FX_INSTRUMENT_NOT_UNIQUE`) in the new `fx` gate and setup step.
- A subscription POSTed across a reconnect is deleted, not registered. A retired strike's rolling
  window yields its slot to a live contract.
- A decision-worker error is the sticky, alerted `fatal_error` and blocks arming; identical broker
  evidence is audited once; internal fills are labelled `INTERNALLY_SIMULATED_FILL` inside the fill
  transaction; reconciled open trades read `OPEN`.

## Recorder and dashboard (`7857f98`)

- Queue pressure ends only the affected capture and its marker follows once the writer drains; the
  recorder no longer switches itself off until restart. Write failures log a coded line.
- Rows carry their sequence, so bridging prefixes decode only what they send; rows-only writes no
  longer copy and fsync the manifest per message; the state-byte accounting and
  `provider_timestamps` are gone. `recorder.pre_event_minutes` (never read) left the
  configuration — see the migration note in [deployment](DEPLOYMENT.md).
- One card builder serves the overview and market pages; the book-history cache actually hits;
  diagnostics load only while the raw-state panel is open; the evidence blob is served once;
  hardening headers (`Content-Security-Policy`, `X-Content-Type-Options`) are on every response.

## Repository trim (`5d88431`), dead code (`8c84494`) and module split (`833084c`)

The repository now holds only what runs on or ships to the server: `stocker_execution` (entry
point `stocker futures-run`, an argparse module) and `stocker_dashboard`, their 185 tests, the
operational scripts and the deployment evidence. Runtime dependencies are pydantic, pyyaml,
httpx, fastapi, starlette, uvicorn and websockets; the locked development environment has 42
packages instead of 184 and the suite runs in about 11 s instead of 69 s. `saxo_data` keeps the
stream, subscriptions and option lifecycle; `saxo_balance` and `saxo_history` hold the account view
and the bars/reference sessions.

## Behaviour an operator should know

- Archive rows no longer carry `provider_timestamps` (the verbatim provider message is still
  recorded) and a row dropped while gapped is flagged `dropped`, not `duplicate`.
- An unreported chart delay blocks history as `SAXO_CHART_DELAY_UNKNOWN`, matching the quote rule.
- Unmapped Saxo order statuses stay non-terminal; a stuck order still surfaces through
  `PENDING_ORDER_RECONCILIATION_REQUIRED`. No Saxo status enumeration was invented.
- The git history still carries the removed research artifacts; rewriting it is a separate decision.

## Checks executed

Ruff lint and format, mypy strict (27 source files), 185 Python tests, the locked server-only
install smoke, the recorder and API benchmarks recorded in [the cleanup document](SLRNO-CLEANUP.md),
and a 4-tick × 400-price sweep of the grid arithmetic (no off-grid result). The Playwright suites
were not run locally (no Node on the workstation); CI's `frontend` job runs them.
