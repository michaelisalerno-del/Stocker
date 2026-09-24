# FIRST4 PAPER cutover — 23 September 2026

Operational code revision: `45bb4d85862241aafb7220d2f6b47b07e8c20f41`.
Installed with the existing release-directory/locked-uv/systemd deployment mechanism on `139.59.178.164`; `/opt/stocker/current` points to that revision. `stocker-v1.service` now invokes `first4-run` with `/etc/stocker/v1/first4.yaml` and `/var/lib/stocker/v1/first4.sqlite3`. No GitHub deployment was assumed.

Before stopping the old manager, all four UK LSE, Korean KRX, US ALL and Australian ASX runs were disabled. The active manager reported zero entries, zero market-data subscriptions/scanners, and reconciled PAPER account `DUP655399` with zero open orders and positions. An independent connection through the newly installed broker code confirmed the same account and zero exposure, with both order-submission methods explicitly blocked for the probe.

The old manager was stopped cleanly. All active old runs and PAPER/LIVE connection configuration were moved out of `/etc/stocker/v1`. Its runtime database, prior run file and consistent SQLite backup were preserved in `/var/lib/stocker/backups/first4-45bb4d858622` (root-only). FIRST4 uses its own ledger and cannot parse or restore the legacy runs. The service definition no longer contains the retired command. Installed old CLI commands fail, old API routes return 404, and the active release contains neither the legacy execution modules nor packaged method models.

The independent three-scanner observer's service/timer and collector hashes are unchanged. Its timer remains active. Its historical pinned release remains available as its dependency; historical releases have not been claimed erased. Gateway PID `1211511` was unchanged. After an additional unarmed restart, FIRST4 reconnected and reconciled successfully; its only running trading process is the US FIRST4 service. See [captured deployment evidence](first4-deployment-evidence.json).

The installation occurred after the US close. Today's scanner history is correctly marked missed; the app will begin a fresh session at the next US opening. This check did not demonstrate intraday scanner operation, executable option quotes, or a broker fill. No FIRST4 order was submitted and no FIRST4 fill was observed. Actual broker submission code is installed, but fill and quote handling have only been exercised with focused test doubles. No synthetic valuations or fills are used by the operational app.

## Concrete prerequisite for arming

The authoritative synthetic artifacts do not approve real listed-option mappings or an executable premium budget. All nine fields below remain null in the deployed config; `armed` is false and LIVE is unavailable:

| Setting | Required choice |
|---|---|
| `expiry_rule` | `EXACT_CALENDAR_DATE` or `FIRST_ON_OR_AFTER` the frozen two-day date |
| `strike_rule` | `OUTWARD` or `NEAREST_TIES_OUTWARD` from .98/1.02 baseline references |
| `premium_budget_usd` | Positive budget per selected opportunity |
| `fee_reserve_per_package_usd` | Nonnegative reserve for each actual option package |
| `entry_limit` | Explicit acceptance of `SUM_OF_ASKS` debit, rounded down to the broker tick |
| `quote_max_age_seconds` | Maximum permissible age of each actual leg bid/ask |
| `entry_deadline_seconds` | Positive baseline execution window shorter than 60 seconds |
| `exit_seconds_before_close` | Explicit pre-close submission offset, 1–300 seconds |
| `exit_order` | Explicit acceptance of `MARKET` close-out |

Populate these values from approved conventions, set `armed: true`, and restart `stocker-v1` before the next US session's first scanner minute. Mid-session restart cannot reconstruct missed first appearances and therefore blocks new admissions for that session. Choosing mappings, quotes, budget or pre-close timing is an execution difference, not a change to the frozen Q5/FIRST4 selection or permission for extra delay. The historical £100 fractional illustration is not used as a live premium budget.

PRIOR15, ordering, deduplication and baseline timing are specified in [FIRST4.md](FIRST4.md), with preserved source/protocol hashes under `tests/fixtures/first4`. Broker-response order never changes slot allocation, and failed execution consumes the original slot.

## Verification completed

- Final full Python suite: 403 passed, five warnings (one dependency deprecation and four existing empty-slice warnings).
- Focused FIRST4/security run: 20 passed, including all 1,293 frozen first appearances and 80 FIRST4 selections across 20 sessions; timing, boundaries, deduplication, permanent allocation, account isolation, partial leg exits, reconnect identity and actual combination metadata request shape.
- Ruff lint and format checks passed; mypy passed for all 106 source files.
- Playwright dashboard checks passed on desktop/mobile, including escaping broker data and removal of legacy controls.
- Locked server-only install smoke passed locally and on the deployed server as the service user.
- Real PAPER connection/reconciliation and unarmed service restart passed. Actual contract quotes, order acceptance and fills require genuine eligible opportunities after the execution settings are supplied; no test trade was placed.
