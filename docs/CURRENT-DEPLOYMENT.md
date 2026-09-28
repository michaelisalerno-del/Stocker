# Runtime evidence status — Saxo migration

**Saxo cutover has not been performed or verified.** No authenticated Saxo account, data entitlement,
L2 feed, recording licence or production server connection was available in this task.
No execution manager, Gateway, service, timer, subscription account or financial account was changed.

The working directory initially contained no application. The clean existing checkout at
`../2026-09-27-task-replace-first4-completely-with-the` was copied with Git into this workspace,
from branch `codex/futures-paper-replacement`, commit
`8f22d2b6e052d63ec440410c7ecb8e2f66b432d8`. The original checkout remains unchanged.
The migration branch is `codex/saxo-only`; its origin is that local checkout, not a publishing remote.

The previous repository report identified server `139.59.178.164`, immutable release
`3ac8cf32afcc78d602bf1041d79b72fa97e99212`, `/opt/stocker/current`, and an armed IBKR PAPER
runtime whose markets were blocked. Its last cited read-only flat snapshot was 2026-09-28 04:32 UTC.
Those are **dated repository observations**, not current server verification or permission to stop
position management. The original report is preserved byte-for-byte at
`research/operational-history/parked-ibkr-runtime/docs__CURRENT-DEPLOYMENT.md.txt`.

An authorised SSH host/user is still needed to inspect actual units, cron/supervisor jobs, restart
paths and current obligations. Follow [the cutover guide](DEPLOYMENT.md); defer stopping any manager
if positions, orders, partial fills or other obligations cannot be accounted for.
