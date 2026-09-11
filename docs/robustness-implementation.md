# Robustness implementation — 2026-09-11

## Identity and scope

Started from reviewed main `fcb2c73ef4b8e05ba9d1c24b8ff78b3bc4d186a8`.
The task directory was empty; a dedicated worktree was attached at
`/Users/michaelsalerno/Documents/Codex/2026-09-11-task-implement-the-stocker-robustness-and`,
branch `fix/robustness-professionalisation-20260911`. Existing checkouts/work were preserved.

Latest fetched main is `c417e1c266a5fb6270ce3aee5687e2fc68d97781`.
Its scanner scheduling fix is incorporated as `51823bf815236272e17596dd498eb1c2dc8c1d48`;
`git cherry origin/main HEAD` confirms patch equivalence. No merge or force push.
The user subsequently authorized incorporation and deployment, superseding the original
deployment prohibition. Broker accounts, Gateway operation and real/manual orders remain excluded.

Reviewable increments:
- `2487518`: mechanical formatting of existing files.
- `51823bf`: user's concurrent scanner scheduling change.
- `ffca54f`: existing lint/type failures; no research retuning.
- `dfc410d`: shared admission, runtime/persistence, dashboard security/freshness,
  diagnostics and regression/release tooling.
- Documentation/release evidence follows separately.

## Checklist and findings

- [x] Baseline established; existing failures distinguished from regressions.
- [x] Atomic pending entry exposure, qualified identity, partial-fill accounting and restart protection.
- [x] Explicit exposure policy and broker credit checks, with explainable admitted quantities.
- [x] Off-lock run preparation, revision validation and responsive pause.
- [x] New identity persistence before activation; explicit failure/retry/restart behavior.
- [x] Local/protected dashboard boundary, proxy assumptions and startup failure isolation.
- [x] Selected/history/causal/evaluated/skipped coverage diagnostics.
- [x] Independent panel freshness, bounded reads, uncertain mutations and percent inputs.
- [x] Independently reporting checks and isolated locked server smoke.
- [x] Consistent SQLite WAL snapshot and disconnected restore regression.
- [x] README, architecture and operations documentation updated.
- [x] Remote CI and authorized deployment verified (actual outcome below).

| Finding | Disposition and implementation | Regression evidence |
|---|---|---|
| Pending entries bypass capacity | Fixed in `ExecutionLedger.reserve`, `active_records`, `Stage7ExecutionService.execute`; immediate SQLite transaction, account/environment/conId scope, stricter active limits, ledger revision validation. | `test_stage7_ledger.py`, `test_robustness_admission.py`: unfilled/cross-run/same-conId/partial/cancellation/uncertain/restart cases. |
| Risk sizing lacks funding/exposure bound | Confirmed risk formula alone was insufficient. Fixed through explicit `RunRiskConfig.max_gross_notional`, fresh IBKR summary, final-quantity what-if credit preview and recorded sizing reason. BuyingPower was not treated as cash. | `test_stage7_execution.py`, `test_stage7_ibkr.py`: narrow stops, unchanged ordinary trade, invalid/mixed currency, missing fields, unset sentinel, credit failure and concurrent preview. |
| Slow preparation blocks runtime/control | Fixed hot `apply_runs_config`/preparation split, revision checks, `pause_new_entries` plus command pause revisions. `minimum_tick` is bounded and cancelled waiters cleaned. | Event-driven `test_stage8_runtime.py`: other run cycles, late prep cannot undo pause, held scheduler/command locks, queued older enable rejected; later deliberate enable succeeds. |
| Unsaved new run can remain active | Confirmed. `RunControlService.add_universe_run` now saves identity before activation; retries recognize unprepared/degraded saved identities. | First-write failure never activates; activation failure remains saved; isolated standalone restart retains identity; retry after partial activation reuses it. |
| Dashboard boundary unclear | Repository lacked explicit backend protection. Actual server preflight found Caddy Basic auth/TLS and loopback backend. Added local, direct Basic and authenticated-loopback-proxy modes; no identity header trust. | `test_robustness_operations.py`: remote/local, unauthorized controls, Origin, forged forwarded/user headers, private proxy token. Integrated failed-ASGI-startup regression preserves engine. |
| Candidate count implies causal coverage | Concern confirmed: adapter tick limit is 5% of configured quote lines; 100 lines means five local tick streams. Added observed counts/reasons without changing selection. | `test_ibkr_resources.py`: budgets 100/600 against 30 selected, per-stream broker rejection. Method data readiness includes exact required context history; consumed stream evidence retained. |
| Stale tables/hung polling/ambiguous controls | Fixed common `api`, panel replacement/stamps, page revisions, read aborts, unknown mutation result, percent conversion, pause wording and saved/applied/activation-failed status. | Three Playwright scripts, including candidates/trades/settings form preservation, response ordering, timeout/recovery and uncertain mutation. API safe-error regression verifies no exception secret leakage. |
| CI stops at first failing tool | Confirmed baseline failure. Independent six-job CI matrix and aggregating `scripts/check.sh`; all jobs must pass. | Check script runs all six; CI smoke asserts no continue-on-error. |
| Lean installation unclear | Locked server-only installation verified locally in a fresh temporary environment: 60 packages, required frozen model, offline dashboard assets/startup, no pytest/Jupyter. | `scripts/server_smoke.py`; installation selection and service command documented. Genuine model dependencies retained; pyproject/uv.lock unchanged. |
| Front door/recovery/build visibility | README rewritten, earlier research walkthroughs moved to existing research guide, historical deployment record moved to runbook. System exposes observed code/config/method revisions. | Offline restore test preserves committed WAL data, excludes uncommitted row, checks checksums, same saved identity and unresolved signal reservation. |
| Frozen method/market boundaries | Already correct and preserved. No new strategy, selection retuning, geometry/exit change, market or trading hours. | Existing frozen/candidate/causal/payoff/exit tests plus identity comparison below. |

## Commands and results

Commands are run through the required `rtk` wrapper. Normal check definitions are
in `scripts/check.sh`; the following are their exact substantive commands.

Baseline:
- `uv sync --locked --all-groups`: passed, Python 3.12.13.
- `uv run --no-sync ruff format --check .`: failed, 78 files.
- `uv run --no-sync ruff check .`: failed, 47 findings.
- `uv run --no-sync mypy packages apps`: failed, 135 errors in 26 files.
- `uv run --no-sync pytest`: 1,031 passed, 14 skipped, 10 warnings.

During implementation:
- First full regression: 1,037 passed, 14 skipped, four failures. These exposed two
  fixtures assuming conflicting duplicate admissions and two adapter fixtures missing
  required currency. Fixtures were corrected without weakening the asserted behavior.
- Subsequent focused run: 91 passed, two failures (narrow-stop quote crossed its stop,
  and audit assertion read a different temporary database); corrected fixtures.
- Combined execution/scanner regression: 119 passed.
- Latest combined ledger/execution/IBKR/runtime/dashboard/resource/restore/method regression:
  213 passed, five existing warnings.
- `npm test`: all three Playwright scripts passed. Chromium installed through the
  locked Playwright dependency. Node was supplied from the desktop's bundled runtime.
- `uv run --no-sync python scripts/server_smoke.py`: passed in a clean temporary
  server-only environment, 60 packages, network prohibited during runtime smoke.
- Final `bash scripts/check.sh` at `6ff4a9d`: all six checks passed, including the
  isolated server install and offline smoke. Python: **1,066 passed, 14 skipped,
  10 existing warnings**, 92.78 seconds. Format: 269 files; typing: 142 source files.
- `git diff --check`: passed.
- `git fetch origin main`: passed; latest main c417e1c included by equivalent cherry-pick.

The 14 skips are existing opt-in broker/integration tests; no real broker is required
or contacted by normal checks. Existing warnings are the Starlette/httpx deprecation,
research empty-slice warnings and discontinued calendar-break metadata.
A host resource-exhaustion interval blocked tools earlier; work resumed after the user
recovered the host. Sandbox SSH restrictions were retried with granted network access.
A local Git identity comparison emitted nonfatal macOS temporary-cache permission warnings;
it completed successfully and the actual content comparisons are recorded below.

## Frozen identity evidence

All six files under `stocker_core/method_artifacts/session_hard` are byte-identical
to the reviewed baseline. MODEL_T0 SHA-256:
`68c0f5ebf4a23744a171e32a32a8b336e8d832013692913ca034f744a88e8bc2`.
The fixed specification hash remains
`fe9e198cc8c31b67bf7ac7da1a12c0264e8512d3741c49609369f156ba9d56f5`.

Across tracked method artifacts, fixtures and research records, 38 files are
byte-identical; the only different file is the mechanically formatted historical
`research/realized_m_20_ibkr_fast_v0/run_experiment.py`, whose Python AST is identical.
AST comparisons also confirm unchanged `methods.py`, `markets.py`, `strategies.py`
and `session_hard_method.py`. Context metadata additions report history readiness
without changing the method's calculations. `pyproject.toml` and `uv.lock` are unchanged.

## Migration and material limits

Existing saved runs and history remain readable. Ledger schema additions are nullable;
old sizing metadata remains unknown. New entries require an explicit gross notional
ceiling; none was invented for existing configurations. The final quantity also requires
verified account/instrument currency compatibility and a valid IBKR credit preview.
One unresolved entry at a time per account/environment is an explicit conservative
admission policy because broker credit cannot be attributed to working parents.
Historical trade counts/performance are not newly validated under that policy.

The current method remains PAPER-only. Run preparation/READY is not proof of complete
candidate data coverage or valid exposure policy; cards expose entry-policy blocking.
Configured feed budget is not entitlement evidence. Existing missed-window and opening
acquisition failures remain historical facts and cannot be repaired by redeployment.

Backup/restore tests are isolated and disconnected. They do not verify the actual server's
backup schedule or offsite retention. Main-branch protections remain an operator action.
No GitHub administrative setting was changed. Actual deployment verification is recorded
below, separately from local tests.

## Remote CI portability correction

CI run 34627171018 on 6bee84b independently passed format, lint, typing, frontend
and server installation. Python reported 1,064 passed, 14 skipped and one failure:
the existing diagnostic-confirmation assertion matched raw ANSI-colored output.
The failure was reproduced locally by explicitly forcing colored Typer output.
The test now checks the same complete messages after Click ANSI normalization and
covers both plain and colored output (four focused tests passed). Order-confirmation
requirements and assertions remain intact. No runtime code changed for this correction.

The server preparation at 6bee84b used a separate fresh locked environment and passed
offline model/dashboard checks as both root and the actual stocker service user.
All 397 existing deployed files matched c417e1c, with no server-only application files.
At that preparation point the active service remained unchanged; final evidence follows.

## Verified release checks

[GitHub CI run 34627755643](https://github.com/michaelisalerno-del/Stocker/actions/runs/34627755643)
passed all six independent jobs on `6ff4a9d885ee0edffa3b363cf9187ec5416f2bfc`:
format, lint, typing, Python, frontend and server installation. Remote Python results:
**1,066 passed, 14 skipped, 10 warnings**, 226.81 seconds. These are actual remote
results, separate from the local run above. The diagnostic correction adds a colored
output case; it does not alter application code.

The exact `git archive` release SHA-256 was
`961242ef82b69d3e5ba4293945017a2d75163a9fada733149cdde17ad8b6b204`.
On the server, `uv sync --locked --no-default-groups --group server` created its own
environment, and `scripts/server_smoke.py --installed` passed as both root and the
`stocker` service user. The smoke forbids network connections and submits no orders.
Immediately before deployment, `git fetch origin main` still resolved to c417e1c.

## Actual deployment and handover

At 2026-09-11 17:34:43 UTC the application had switched successfully from c417e1c to
**`6ff4a9d885ee0edffa3b363cf9187ec5416f2bfc`**, through
`/opt/stocker/current` on the existing server. PAPER reconciled within 95.1 seconds.
The final documentation commit records this evidence; deployed application code stays
at the CI-verified revision above. The branch is
`fix/robustness-professionalisation-20260911`; main was not merged or force-pushed.

The application service was stopped for the consistent backup and restarted on the
prepared release. Caddy was validated and gracefully reloaded. The Gateway process ID
did not change. There were no open positions, unresolved entry plans or orders today
at cutover, and no orders were placed by this task.

Verified before/after:
- Runs and IBKR configuration files remained byte-identical; all 27 saved identities
  were retained. The three enabled runs retain their original settings.
- Ledger/history counts remained: one execution plan, 28 fills, one execution attempt,
  54,502 runtime signals and zero opening candidate stage rows. Candidate session
  records were unchanged. The sole existing execution plan was already closed.
- The System API reports the exact deployed commit, with dirty state honestly unknown
  for an archive installation. Configuration revision:
  `6ddf8eccb4cb4ce3935ce5afb8dfacf105c0d71778cbbf65ce630efe80f33df0`.
- Authenticated backend reads and all three dashboard static assets succeeded; asset
  bytes matched the prepared release. Public unauthenticated page/System/control probes
  returned 401. Direct backend equivalents and forged identity headers returned 403.
  Authenticated requests with a foreign Origin or cross-site metadata returned 403.
- An external connection attempt to backend port 8765 timed out (curl exit 28).
  This is one observed probe, not proof of every possible network path.
- Existing Caddy Basic credentials and TLS configuration were preserved. A private,
  loopback-only upstream token now prevents bypassing Caddy authentication. Its value
  is held only in restricted server configuration, never in JavaScript or this report.

The private verified snapshot is at
`/var/lib/stocker/backups/robustness-6ff4a9d/state`; deployment and HTTP verification
reports are in its parent directory. It contains a SQLite online-backup snapshot,
matching configuration and frozen artifacts with checksums. The isolated WAL restore
test described above is distinct from this real snapshot verification. No production
database restore was performed, and no backup schedule or offsite retention was proved.

Exact operational entry commands, run through `rtk`:
```sh
ssh -o ConnectTimeout=10 root@139.59.178.164 python3 - < .stocker/prepare-release.py
ssh -o ConnectTimeout=10 root@139.59.178.164 'runuser -u stocker -- /opt/stocker/releases/6ff4a9d885ee0edffa3b363cf9187ec5416f2bfc/.venv/bin/python /opt/stocker/releases/6ff4a9d885ee0edffa3b363cf9187ec5416f2bfc/scripts/server_smoke.py --installed'
ssh -o ConnectTimeout=10 root@139.59.178.164 python3 - < .stocker/deploy-release.py
ssh -o ConnectTimeout=10 root@139.59.178.164 python3 - < .stocker/verify-deployment.py
gh run view 34627755643 --repo michaelisalerno-del/Stocker --json status,conclusion,jobs
curl --connect-timeout 3 --max-time 5 -sS -o /dev/null -w '%{http_code}\n' http://139.59.178.164:8765/
```
The one-use preparation/deployment/probe scripts and copied non-secret verification
reports remain in the ignored local `.stocker` task directory. The reusable install,
backup and recovery procedures are tracked in `scripts/` and the recovery runbook.

Material limits and operator actions:
- **New entries are blocked** for all three enabled runs until the operator sets an
  explicit `risk.max_gross_notional`; no financial ceiling was invented. Saved enabled
  status and a READY method do not override this gate. New IBKR credit admission has
  fake-broker test evidence, but was not exercised by submitting an actual entry.
- Existing US `BROAD_OPENING_DATA_CAPACITY_UNRESOLVED` and UK
  `CANDIDATE_SELECTION_WINDOW_MISSED` states remain. Australia is READY. Redeployment
  did not reconstruct missed session evidence or rerun selection.
- The observed server configuration is 100 quote lines and five tick-by-tick slots;
  actual entitlement remains unknown. Zero streams were active at verification.
- The operator should verify their authenticated browser login, review network access
  paths and backup retention, and configure all six CI checks as required branch
  protections. No administrative settings were changed. Main remains c417e1c; this
  release is deployed directly from the published task branch.

Implementation entry points: `packages/stocker_execution/src/stocker_execution/`
contains `execution_ledger.py`, `stage7.py`, `ibkr.py` and `runtime.py`;
`packages/stocker_dashboard/src/stocker_dashboard/` contains `controls.py`, `app.py`,
`security.py` and `static/dashboard.js`; the configuration schema is
`packages/stocker_core/src/stocker_core/runs.py`. The finding table above maps their
changed functions to focused regressions. Optional later ideas are outside this release.
