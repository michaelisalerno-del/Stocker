# FIRST4 deployment integration — September 24, 2026

After the repair at `d6fd1c099fb151bfc2f21e067852e91a5bc65681` passed all six
GitHub checks, deployment inspection found the server running a separate branch,
`a8fe377e6ae64d0e40cf5f90512c1f8ccc7dc53d`. The original deployment document's
release pointer was therefore historical. The running service was left unchanged
while the differences were reconciled. The user subsequently authorized restart
and checking readiness for today's existing dated opening permission.

This integration retains the active release's durable IBKR 10197 block and
positive option-access recovery, strict account/client/order/permanent identities,
stale status rejection, connection-generation checks, conflicting exit-order
protection, fresh reconciliation before final closure, working-exit visibility,
changed-only position writes, and view-specific bounded dashboard queries.
The existing research fetch retirement and its test/documentation were copied
unchanged from that release; no research experiment was run. Its two historical
reliability reports are preserved as evidence, not as claims about this patch.

The supervised runtime, correction accounting, targeted obligation queries,
calendar validity, frozen-appearance handling and narrower ambiguous-expiry
handling from the new repair remain. Active-release payload completion markers
are not assumed authoritative by the additive migration: obligations start active
until reconciliation proves completion. No runtime block or dated permission is
cleared by deployment.

Local verification on the integrated source:

- `rtk bash scripts/check.sh python`: 487 passed, five existing warnings, 62.99s.
- `rtk bash scripts/check.sh typing`: 108 source files passed.
- `rtk bash scripts/check.sh format`: 187 files formatted correctly.
- `rtk bash scripts/check.sh lint`: passed.
- `rtk bash scripts/check.sh server`: locked server-only offline smoke passed.
- `bash scripts/check.sh frontend`, through the same bundled Node/npm runner
  documented in FIRST4-repair-verification.md: passed.
- `rtk git diff --check`: passed. Frozen source/fixtures, configuration example
  and dependency locks have no diff.

`tests/test_first4_deployment.py` covers persistence of the competing-session
block, failed reporting, wrong order/execution identities, stale statuses,
interrupted reconciliation/chain requests, working and conflicting exits,
unchanged position writes, bounded view queries and retained ledger metadata.
The existing fake executions now include the account/client/order/permanent
identities required by the actual API. No safety assertion was removed.

Pre-restart read-only inspection found today's opening gate WAITING_FOR_OPEN,
no session block, no recorded orders or positions, and no active 10197 block.
Only one Gateway process and one FIRST4 API connection were present. Client 181
was not connected. The independent observer was inactive with its existing
13:29 UTC timer; it was not modified.

This integration record does not assert activation or successful opening quotes.
Actual activation evidence belongs in CURRENT-DEPLOYMENT.md after verification.
Premarket service availability is not proof that the 13:30 UTC opening check will
pass. No additional date, manual arming, test trade or Gateway restart is included.
