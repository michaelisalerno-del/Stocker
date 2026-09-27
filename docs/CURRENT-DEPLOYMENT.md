# Implementation and deployment status

Implementation base: `86d4790eb142f85a75792546dd19da4512e05a9e` (GitHub main).
Working branch: `codex/futures-paper-replacement`.

**Futures replacement is local, unarmed and not deployed.** All six markets remain visible;
all six default to orders blocked pending listed-product/expiry/delta-tolerance authority.
The prior production release remains active. No server configuration or schedule was changed.

Fresh read-only IBKR snapshot at **2026-09-27 09:20:13.033138 UTC**:
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
