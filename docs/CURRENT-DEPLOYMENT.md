# Implementation and deployment status

Implementation base: `86d4790eb142f85a75792546dd19da4512e05a9e` (GitHub main).
Working branch: `codex/futures-paper-replacement`.
Market-data/L2 extension base: `4eff5fc3b55f703bd78cf1794a25561847413f9b` (local futures migration).

**Futures replacement is local, unarmed and not deployed.** All six markets remain visible;
all six default to orders blocked pending listed-product/expiry/delta-tolerance authority.
The prior production release remains active. No server configuration or schedule was changed.

Earlier migration-stage read-only IBKR snapshot at **2026-09-27 09:20:13.033138 UTC**:
- Broker-returned managed account exactly `DUP655399`.
- Zero open orders, zero nonzero positions, zero returned executions.
- Read-only client 89; zero orders transmitted. Disconnected after the check.

Authenticated existing app read earlier in this session reported zero outstanding obligations,
connected/reconciled and unarmed. These observations do not guarantee flatness at a later cutover.
The active release observed through systemd/readlink was a7ac6db9c81cb140af857d2af5534e397153785e,
PID 1392170. Historical deployment reports are archived separately and do not describe current state.

Remaining operations: review implementation; approve/stage coordinated cutover while flat;
verify deployed data/metadata; explicitly approve listed execution mappings and then arming.
Actual FOP acknowledgements, fills, cancellation races, commissions and exercise handling are
covered with offline fixtures, not verified through transmitting test orders.

The market-data/L2 addition is also local and awaiting deployment. It made no broker connection,
changed no remote service or collector, and transmitted no orders. Its account allowance is
ASSUMED, external usage unknown, L2 disabled and execution unarmed. Documentation verifies IBKR
API semantics, not current account entitlements. Repeat the non-transmitting account and data
preflight at cutover; none of the earlier observations proves present flatness or available capacity.
See [market-data operating behaviour and fixture screenshots](MARKET-DATA.md).
