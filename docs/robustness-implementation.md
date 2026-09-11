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

## Post-deployment numerical reproducibility correction

The documentation-only CI run 34629079092 and diagnostic run 34629694002 each
reported **1,063 passed, 14 skipped and three failed** exact frozen-score tests.
Both application code and those tests were unchanged from the earlier green run.
The first differences were one ULP in RV scores for ALL, AES and BAX. The failing
runner's NumPy 2.4.6 diagnostics selected `X86_V4` for float64 `log`; the existing
server has V2/V3 support and reproduced all **4,782** frozen scores and all three-stage
watchlists exactly in a disconnected process using a separate prepared environment.
The Mac also reproduced the exact scores. No broker connection was used by these probes.

`stocker_launcher.configure_numeric_runtime` now makes the verified numerical path
explicit before console application imports. On x86 it retains any existing CPU
restrictions and excludes NumPy's `X86_V4`, `AVX512_ICL` and `AVX512_SPR` dispatch
targets; ARM remains unchanged. Pytest initialization and server smoke call the same
helper. This is a shared application startup setting, not a test-only relaxation.
The frozen candidate function, all exact equality/order/watchlist assertions, model
and fixture files remain unchanged. This preserves the established numerical path
on newer hardware rather than accepting altered scores. NumPy documents its
[CPU dispatch environment setting](https://numpy.org/doc/2.4/reference/global_state.html).

Both console-entrypoint regressions failed before the helper was added; they now
verify that the profile applies before application startup and preserves operator
restrictions. An ARM regression verifies its environment remains unchanged. The
existing exact-score suite supplies the real numerical regression on CI hardware.
An initial local check found the installed force-included launcher still had the old
contents; `uv sync --locked --all-groups --reinstall-package stocker` rebuilt it.
A direct launcher mypy check also exposed an existing importlib path-protocol typing
issue; converting its filesystem path through `str` resolves that without changing
the package-location behavior. No dependency or lockfile change was needed.

The 6ff4a9d deployment above is the first successful cutover record. Subsequent
numerical-profile release identity and its final CI/deployment outcome are recorded
in the task handover and the server's System surface and private verification reports.
The profile does not change the numerical path on the existing server's V2/V3 CPU.

CI run 34630257236 verified the numerical correction on an X86_V4-capable runner:
all exact frozen-score cases passed, with diagnostics showing the application profile
selecting X86_V3 for float64 log. The run reported 1,067 passed, 14 skipped and two
different failures in the incorporated scanner regression. Its fake broker assumed
natural request overlap despite independent SQLite thread scheduling, and the
three-sweep case exceeded a two-second whole-test watchdog. The test now uses a
two-party barrier to require actual concurrent scanner calls and a 30-second deadlock
watchdog. All existing pool, request, qualification, concurrency and persisted-state
assertions remain exact. Qualification still cannot finish until every requested sweep
arrives, so restoring the original blocking defect would still deadlock and fail.
No scanner implementation or method deadline changed for this test correction.

## Follow-up: slow dashboard and delayed pause acknowledgement

The user reported this on deployed `d392d83230bc1c98251266d585cd3cb032f023ef`.
Read-only probes confirmed Australia was saved disabled and subsequently reported
`DISABLED / New entries paused` by the running service, with zero positions and
zero orders that day. During the command, both a static CSS read and overview
exceeded 12 seconds. A stack-only `py-spy dump` found the main asyncio thread in
`yaml.safe_load`, called by the dashboard's `changed()` response refresh. The
8.9 MB configuration includes historical snapshots; these are retained.

The control path also synchronously parsed configuration before its pause gate
and synchronously serialized the save. Three HTTP regression cases reproduced
these defects before the fix. `disable_run` now gates the runtime before any disk
read; asynchronous controls and response refreshes offload large YAML work.
An independent storage lock serializes writes without queuing pause behind broker
preparation. Pause reads after prior writes, preserving concurrently saved run
identities. A cancelled writer retains the storage lock until its worker finishes.
Responses reuse the exact completed save snapshot instead of reparsing the large
file; its monotonic write revision prevents a late response publishing older state.
Commands that did not write still load off the runtime thread under a refresh lock.
PyYAML's `CSafeLoader`/`CSafeDumper` retain safe construction and atomic replacement
while avoiding the slow pure-Python parser/emitter. Existing saved schemas and
all frozen artifacts remain unchanged; a subsequent save may change YAML formatting.

The six new regressions cover held reads, writes and response serialization with
concurrent HTTP access, and concurrent identity persistence with/without writer
cancellation, plus out-of-order result publication. The dashboard, runtime and
configuration focused suites pass. The initial isolated server benchmark preserved
all 27 identities and every configuration value except its requested temporary
pause; it gated entries in 0.0102 seconds but acknowledged in 27.77 seconds. This
exposed the redundant post-save read removed by the saved-snapshot response.
Full release checks, isolated large-configuration measurements, and deployment
verification are reported with the final release identity in the task handover.

## Follow-up: remaining cycle stalls and slow Enable

On `cd091a8`, a ten-second stack profile found 270 of 314 samples inside
exchange-calendar resolution; each cycle reconstructed the same calendar/day
several times. `ExchangeSessionResolver` now retains the latest schedule per
calendar, while recalculating the current clock state, configured window and slots.
Regressions cover intra-day transitions, changed windows, holidays, half-days and
US daylight saving. Existing market and frozen method calculations are unchanged.

The saved 8.9 MB configuration contained nine identical 7,497-member lists and
three copies of each of two other lists. Storage now shares exactly equal member
lists using standard YAML anchors/aliases. No identities, member ordering, snapshot
values or provenance are removed. Both the old and new representation load through
the existing schema; typed snapshots are independent after loading. Atomic writes
remain in place. Operators should edit risk and run controls through the dashboard,
and should not manually edit frozen membership anchors. Code rollback can read the
new YAML representation without a database migration.

The browser immediately labels pending controls, suppresses duplicate operations
across panel refreshes, and leaves Pause available while Enable is pending. API
timeout uncertainty and broker readiness remain distinct from a saved change.
Both Python regressions and the browser pending-state regression failed before
their fixes and passed afterward. Release validation includes an offline copy of
the real saved configuration and a disconnected enable/pause benchmark.

## Continuation acceptance

This section supersedes the earlier incomplete handover. The continuation found the
original task worktree already clean at **a1680fcdbf0f3b689cac6f9e394ba1bd274e07d9**,
on `fix/robustness-professionalisation-20260911`. The newly created thread directory
was empty; no checkout was recreated and no existing change was reverted.
The complete hardening comparison baseline remains
`fcb2c73ef4b8e05ba9d1c24b8ff78b3bc4d186a8`. Existing 13 commits were retained.
This continuation did not deploy, contact IBKR, alter broker accounts or place orders.

Verified implementation/test commits:

- `85c1c4c461797abc4ec3805bea86f4e1984693f3`: fence stale sizing and retain broker-reported
  fills awaiting execution details, including migration and concurrency regressions.
- `bd9f2d598c365ad28d0b684aef804047995128a5`: complete runtime/security/browser/persistence
  acceptance and opening-to-dashboard rehearsals.
- `cecf5146ca1ec1a909ee39185c7e727df13225f8`: position the header-refresh assertion after the
  panel becomes stale; rerun all three browser scripts successfully.
- The following documentation-only commit records this report; its exact final hash
  is given in the final handover.

### Original workstreams

| Workstream | Final local status |
|---|---|
| Baseline/instructions/current changes | VERIFIED/FIXED — root AGENTS reread; no nested AGENTS or CONTEXT files found; clean starting branch and existing commits inspected. |
| Pending-entry/exposure | VERIFIED/FIXED — existing atomic reservation retained; terminal-status-before-fill gap fixed. |
| Explicit sizing/capacity | VERIFIED/FIXED — existing limits/currency/credit checks retained; stale risk configuration fence added. |
| Runtime/config concurrency | VERIFIED/FIXED — existing off-lock preparation/pause/revision design verified; newer-enabled-config regression added. |
| New-run persistence/retry | ALREADY CORRECT — save-first identity, activation-failure/retry tests inspected; failed-save restart assertion added. |
| Dashboard security | ALREADY CORRECT locally — route inventory and websocket-scope regressions added; current server boundary is EXTERNAL VERIFICATION REQUIRED. |
| Candidate/feed coverage | ALREADY CORRECT — below/sufficient-capacity and individual-failure tests plus integrated 30/30/30 coverage passed. |
| Dashboard freshness/controls | ALREADY CORRECT — added direct header-vs-panel, real polling timeout/recovery and no-mutation-retry assertions. |
| Independent release checks | VERIFIED/FIXED — all six canonical checks passed; isolated Node 22 resolved local missing npm. |
| Locked server installation | ALREADY CORRECT — clean 60-package install, model/import/startup/assets smoke repeated successfully. |
| Documentation/recovery/build identity | VERIFIED/FIXED — current guides updated; existing consistent WAL snapshot/restore regression passed. |
| Scope/frozen behavior/commit audit | VERIFIED/FIXED — original method/selection artifacts unchanged; shared execution changes explicitly recorded. |

No local acceptance item remains blocked. SSE/websocket data routes are NOT APPLICABLE
to the current app (none are defined); middleware's websocket rejection is nevertheless tested.

### Concurrency evidence

`ExecutionLedger.reserve` acquires SQLite `BEGIN IMMEDIATE` before checking signal
uniqueness, revision, account/environment, conId occupancy, the strictest applicable
position/notional limit, and pending credit. Insertion and sizing metadata commit
inside that transaction. Two connections cannot both pass using the same unreserved
slot. `test_concurrent_cross_run_reservations_share_one_slot` uses a thread barrier;
new same-conId barrier cases assert exclusion in the same scope and independent
admission across account/environment boundaries. Local reservations supplement broker
positions. A set union counts a partial fill plus its parent once. Only ENTRY joins
participate; stop/target/timeout children do not consume slots.

The existing service reserves before credit preview; the event-gated two-run preview
test permits only one preview/commitment. Multiple recovered commitments are still
fully accounted for; production admission blocks another unresolved parent because
IBKR preview cannot attribute existing credit. No distributed coordination was added.
The supported process/database boundary is documented in execution_safety.md.

Two additional defects were reproduced before correction:

1. Risk edits during quote or credit awaits allowed the original 100-share sizing
   after risk was reduced tenfold. Both event-driven cases failed, then passed after
   capturing run configuration identity and checking it after the awaits. No await
   separates the final fence, marking SUBMITTING and the concrete IBKR bracket
   transmission (the adapter coroutine has no internal await).
2. FILLED(100), CANCELLED(40) and REJECTED(40) parent status received before execution
   details removed active exposure. All three failed with zero active records.
   A nullable cumulative reported-fill field now keeps unmatched shares reserved.
   Actual positions still require execution records. The report is monotonic across
   repeated/out-of-order statuses; fills arriving later settle only the missing
   amount. Six cases cover current/old schema and restart. A separate service test
   rejects reconciliation despite visible protective orders until the fill arrives.
   Old local pre-submission rejection remains settled after migration.

Existing tests cover unfilled-first/second admission, cancellation/rejection, partial
fill financial remainder, unknown submission/restart, account changes during preview,
wrong PAPER/LIVE destination and protected-order reconciliation.
The account revision now includes cumulative reported fills, so a change during
financial preparation also invalidates its snapshot.

Runtime preparation uses per-run revisions and prepares outside `_cycle_lock`.
Publication rechecks revision/service identity under that lock. The new event-driven
test completes a newer enabled/risk configuration before releasing the old preparation
and proves the manager, executor and strategy still refer to the newer state.
Existing tests hold run A qualification while run B cycles, pause under held command
and scheduler locks, reject queued older enable, and permit a deliberate later enable.
Minimum-tick and execution-state timeout tests assert cancellation cleanup.

The dashboard command lock serializes ordinary mutations; its separate storage lock
retains ownership until a cancelled writer completes. Pause gates synchronously before
disk/broker work and increments a revision/latch. Existing held-writer/cancellation
tests preserve concurrently saved identities. Saved response revisions prevent an
older completed command from publishing stale configuration.

### Security evidence and boundary

**A — repository guarantees.** Every current FastAPI handler, documentation endpoint,
static mount and defensive websocket scope passes DashboardSecurity. Local mode
requires loopback client plus local Host. Protected Basic mode requires the configured
credential and Host; proxy mode additionally requires a loopback peer and private
upstream token. Identity/forwarded headers confer no authority, and both CLI launchers
set `proxy_headers=False`. Default bind is 127.0.0.1.
Origin mismatch/cross-site fetch metadata are rejected, including for browser-sent
Basic credentials. There is no permissive CORS middleware. Mutations use POST/PUT;
the private token/password is not emitted in static JavaScript or API responses.
Exceptions return safe messages with reference IDs; secret-bearing control exception
regressions pass. Invalid dashboard startup remains isolated from engine operation.
Supported remote use requires TLS at an authenticated/private proxy with a loopback
backend, as documented; a local-mode proxy rewriting Host is unsupported.

**B — actual server checks.** This continuation made no remote security probes.
Earlier deployment evidence above is historical, not certification of this final
revision or the current host. Verify service bind/firewall and every reachable backend
path, TLS, proxy authentication/token overwrite, unauthenticated reads/controls,
forged headers/cross-origin rejection, secret-file permissions and authenticated browser
access after an authorized deployment. No general user-management system was introduced.

### Acceptance and opening rehearsal

The full suite now has **1,096 passed, 14 skipped, 10 warnings** (1,077 passed at this
continuation's start). The 19 additional Python cases include two stale-risk races,
three same-instrument scope races, multiple recovered commitments, six callback-order/
schema cases, reconciliation of missing execution details, local-rejection migration,
newer configuration publication, whole-app security, websocket rejection and two
integrated rehearsals. Existing tests were extended for failed-save restart and browser
freshness/polling, without dropping prior assertions.

`tests/test_hardening_rehearsal.py` composes real ScannerAcquisition, CandidatePipeline,
StockerRuntime, Stage5CurrentDataService/PRE calculation, history cache, frozen
SessionHardMethod/MODEL_T0, Stage7 admission, ledger and DashboardReadService.
Only broker/market input seams are test doubles. A pre-existing low-risk frozen model
feature vector supplies valid context; no qualification threshold/model/selection is
patched. Required exact T0/T0-3 bars are supplied through a test IbkrConnection boundary.
Thirty instruments each request the two required history resolutions; PRE is actually
calculated as 0.8M. Explicit test-only expected-move/context fixtures do not establish
live market validity.

Full configured recipe: **3 sweeps × 35 components × 50 rows = 5,250 observations**,
scanner concurrency two, broad saved membership 7,497, qualified unique union 1,750,
3,500 duplicate hits, no failed components, no eligibility/data exclusions.
All 105 components completed. Qualifications are held by an event until every scanner
request has returned; restoring scanner-slot coupling would deadlock the test.
Assumed virtual service charges are 250ms per scanner response, 10ms per unique
qualification and 5ms per prefix request. These are deterministic workload assumptions,
not claimed measured Gateway speeds. They retain the complete recipe/population and
all original deadlines.

Both trade/no-trade rehearsals produced the same virtual opening results (UTC,
2026-09-02; US open 13:30):

| Stage | Input → output | Completed | Required deadline |
|---|---|---|---|
| Scanner acquisition + qualification | 5,250 raw → 1,750 qualified | 13:34:26.250 | 13:35:00 |
| Range5 | 1,750 → 250 | 13:35:08.750 | 13:40:00 |
| RV10 | 250 → 50 | 13:40:01.250 | 13:45:00 |
| RV15 | 50 → 30 | 13:45:00.250 | 14:00:00 |
| Causal stream preparation | 30 selected | 13:55:00 | before T0 14:00 |
| Required history/PRE and evaluation | 30 history-ready → 30 evaluated | 14:00:00 | existing checkpoint/entry rules |
| Observed causal coverage | 30 timely streams with valid prints | 14:00:01 | original causal window |

The trade path produces one eligible SHORT, 100 shares, original 99.6 entry/100.6 stop/
97.6 target geometry, two fake entry fills (40 + 60), protected-position reconciliation,
target exit and one dashboard closed trade. The no-break path keeps all 30 valid
candidates untriggered with zero orders, positions, trades or reservations.
No feeds are rotated after deadline; prints, not substitute OHLC bars, drive entry.
Other existing regressions cover feed budget 100/600, individual subscription failure,
missed windows, qualification timeout/cancellation, and no post-seal writes.
**Fake/local throughput does not prove real IBKR opening-session throughput.**

### Exact check matrix

The final aggregate command ran each required check independently and exited zero:

```sh
rtk env PATH=/tmp/stocker-hardening-npm/node-v22.14.0-darwin-arm64/bin:/Users/michaelsalerno/.local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin bash scripts/check.sh
```

These are the exact child commands selected by that canonical script:

| Command | Result |
|---|---|
| `uv run --no-sync ruff format --check .` | PASS — 272 Python files |
| `uv run --no-sync ruff check .` | PASS |
| `uv run --no-sync mypy packages apps` | PASS — 142 source files |
| `uv run --no-sync pytest` | PASS — 1,096 passed, 14 existing opt-in skips, 10 warnings; 106.12s |
| `npm test` | PASS — all three Playwright/Chromium scripts under Node 22.14.0; repeated after final stale-panel assertion refinement |
| `uv run --no-sync python scripts/server_smoke.py` | PASS — isolated locked 60-package server install, imports/model/offline ASGI startup/assets |
| Server child: `uv sync --locked --no-default-groups --group server` | PASS — new temporary environment, no dev/research groups |
| `rtk git diff --check` | PASS |

The final browser-only refinement was checked with
`rtk env PATH=/tmp/stocker-hardening-npm/node-v22.14.0-darwin-arm64/bin:/Users/michaelsalerno/.local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin npm test`
(PASS, all three scripts). Python code was unchanged after the full matrix.

Focused commands also run:

| Command | Result |
|---|---|
| `rtk uv run --no-sync pytest tests/test_robustness_admission.py -k risk_update` | Initially FAIL (2 demonstrated stale-sizing defects); covered by subsequent green combined/full runs |
| `rtk uv run --no-sync pytest tests/test_robustness_admission.py tests/test_stage7_execution.py` | PASS — 31 at that increment |
| `rtk uv run --no-sync pytest tests/test_stage7_ledger.py -k terminal_status_before --tb=short` | Initially FAIL (3 demonstrated early-release defects); covered by subsequent green combined/full runs |
| `rtk uv run --no-sync pytest tests/test_stage7_ledger.py tests/test_stage7_execution.py tests/test_unattended_reconciliation.py tests/test_robustness_admission.py --tb=short` | PASS — 57 at that increment |
| `rtk uv run --no-sync pytest tests/test_hardening_rehearsal.py --tb=short -s` | PASS — both rehearsal paths, metrics printed above |
| `rtk uv run --no-sync pytest tests/test_stage7_ledger.py tests/test_robustness_admission.py tests/test_robustness_operations.py tests/test_stage10_dashboard.py -k 'terminal_status or status_fill_gap or first_run_write or real_dashboard_routes or websocket_boundary' --tb=short` | PASS — 10 targeted cases at that increment |
| `rtk python3 .stocker/continuation-diff-audit.py` | PASS — complete-path AST/function inventory and baseline artifact comparison; one-use audit script retained locally |

Initial `rtk bash scripts/check.sh`: format/lint/typing/Python/server PASS,
frontend FAIL because npm was absent from PATH. A PATH attempt with the bundled Node
also failed because it has no npm; an isolated npm-only package could not infer that
bundle's layout. An official standalone Node 22.14.0 distribution in /tmp resolved
the tooling issue without changing project dependencies. Frontend and then all checks
passed through the canonical script. Initial new test-draft failures were corrected
for canonicalized RunConfig values, required dashboard arguments and paginated response
shapes; these were test-harness errors, separate from the two demonstrated code defects.
Initial focused lint found import ordering/line lengths in added tests; scoped Ruff
format/import fixes resolved them. No pre-existing failures remain hidden.

The 14 skips are existing broker/opt-in cases; they do not establish live IBKR behavior.
Warnings remain the existing Starlette/httpx deprecation, research empty-slice warnings
and calendar discontinued-break metadata. No tests were disabled, assertions weakened,
broad exclusions added or continue-on-error introduced. These results are local;
remote CI for the continuation commits was not requested or claimed.

### Recovery and complete diff audit

The existing WAL restore test ran successfully in the full matrix: committed data was
restored, an uncommitted row excluded, matching configs/artifacts verified, unresolved
SUBMITTING reservation and identity preserved, and checksum corruption rejected.
No real broker was connected. Production backup scheduling/offsite retention and
production restore were not tested by this continuation.

README, execution safety, dashboard security and the existing recovery runbook now
include the configuration fence, status-before-execution semantics, migration and
rollback caution. Existing installation, Market → Method → Run, PAPER-only, US
session/DST, IBKR-only PRE provenance, saved/applied/preparing/ready, pause semantics,
coverage vocabulary and backup procedures were checked and retained.

Complete-path baseline audit found **38 byte-identical artifact/fixture/research files**;
the sole different tracked research artifact-area file is the previously formatted
`research/realized_m_20_ibkr_fast_v0/run_experiment.py`, with identical Python AST.
All six frozen method artifacts match the baseline. MODEL_T0 SHA-256 remains
`68c0f5ebf4a23744a171e32a32a8b336e8d832013692913ca034f744a88e8bc2`.

`methods.py`, `session_hard_method.py`, `stage5.py` and `strategy_factory.py`
are AST-identical to baseline. Candidate ranking code/fixtures and market definitions
are unchanged; Stage7 risk calculation, order-plan geometry and concrete bracket
submission functions are unchanged. SessionHardStructureD adds only coverage metadata.
The context provider adds readiness observations. ExchangeSessionResolver's inherited
schedule cache changes reuse, not calendar/window/checkpoint formulas; dedicated DST,
holiday, half-day and intra-day transition tests pass. The inherited x86 NumPy dispatch
profile preserves exact frozen values; fixtures/thresholds were not rewritten.
`uv.lock` is byte-identical; pyproject's only inherited change is formatting.
Historical research source has inherited explicit lint/type fixes as documented above;
it is not claimed byte-identical. Those existing commits were retained.

Intentional operational changes across the hardening baseline are atomic shared
entry commitments, explicit notional/credit/currency admission, off-lock preparation
and pause/config fences, saved identity/persistence handling, dashboard boundary and
freshness, observed coverage, scanner/qualification scheduling, schedule/configuration
reuse, safe errors, reproducible numerical startup, and release/recovery tooling.
This continuation adds only the two admission fixes and acceptance evidence.
No intentional change was made to signal rules, thresholds, ranking, PRE mathematics,
stop/target geometry, exits, US session timing/DST, PAPER/LIVE routing, PAPER-only
restriction or IBKR production-history provenance.

### Required external actions and material risks

Before a supervised PAPER session:

1. Review/publish the final branch revision through the normal workflow and obtain
   all six remote CI results for that revision. This continuation made local commits
   only; it did not push, merge or deploy.
2. Under separate deployment authorization, prepare the exact release using the locked
   server command, pause and back up consistent state, deploy it, and verify System
   revision plus runtime/database/configuration compatibility and rollback readiness.
3. Verify the current TLS/authenticated proxy, private backend reachability, credential
   file permissions and authenticated browser access as specified above.
4. Set/review explicit account-currency `max_gross_notional`, risk and shared position
   limits; verify PAPER account identity and broker credit-preview behavior. Do not
   infer these from a saved/READY run.
5. Verify actual IBKR history permissions, quote/tick entitlements and timely observed
   coverage for all required selected instruments. Start before the acquisition window
   and record real sweeps, prefix deadlines, causal streams and evaluation counts.
   The historical 100-line/five-tick configuration cannot establish 30-stream coverage.
6. Reconcile all open/completed orders, executions, positions and protective children,
   including unknown submissions/status-fill gaps. Supervise the first eligible PAPER
   entry, partial/terminal callbacks and protective exit without overriding frozen rules.

Material limits are external broker/network behavior and entitlements, asynchronous
uncertain exposure requiring reconciliation, and the explicit conservative admission
policy's effect on realized trade counts. Local acceptance does not establish production
security, production backup scheduling, LIVE readiness, profitability or trading edge.
Optional improvements: none required for this task; broader infrastructure is out of scope.

**LOCAL_HARDENING_COMPLETE_EXTERNAL_VERIFICATION_REQUIRED**
