# Futures PAPER replacement

Base: GitHub main `86d4790eb142f85a75792546dd19da4512e05a9e`.
Implementation branch: `codex/futures-paper-replacement`.

Authorised: implementation, offline testing and non-transmitting preflight.
Not authorised: server cutover/restart, arming, test orders or LIVE.

Source reconciliation: CURRENT_RESEARCH_NG13_ONLY uses the fixed hourly
09:00–16:00 America/New_York primary architectures from FIXED_ARCHITECTURE_SPEC.
BTC/CL calls 0.10 delta; GC/NG/NQ puts 0.10; SI puts 0.20. Only NG 13:00 veto.
Original endpoint is opportunity +60 minutes, never fill +60. G2/GC16 and volume
veto experiments are not approved management. Listed mapping and tolerance were
not frozen by these synthetic studies: default per-market order blocks are required.

Plan: preserve source evidence; replace runtime/ledger and broker boundary; implement
continuous market monitoring and fail-closed mappings; replace dashboard in existing
FastAPI/vanilla JS stack; remove stock strategy wiring; test lifecycle, parity and UI;
independent standards/spec reviews; prepare unarmed deployment and completion report.

Runtime read September 27: existing service active at release a7ac6db9, PID 1392170;
authenticated status connected/reconciled, unarmed, zero reported obligations/positions.
This is the existing application's report, not a fresh broker exposure request.
No service/config/order mutation performed.

Fresh direct IBKR read-only snapshot 2026-09-27 09:20:13.033138 UTC: exactly allowlisted
account DUP655399; no open orders, positions or returned executions. No transmission.

Implemented neutral rules/contracts/broker/runtime/store/request modules and the six-card
dashboard; removed retired runtime, workers, CLI/config/routes and strategy-only tests/scripts.
Preserved 51 historical documents/data/screenshots byte-for-byte with a manifest. Preserved
six exact frozen configurations and calculation-source text snapshots with hashes.

Review fixes: actual completedOrder wire decoding; mutable IB bar tail exclusion; inherited
finite-feature denominators; exit exposure recheck after awaited quote; startup synchronization
serialization; fresh FX admission; same-contract closure capacity release; historical rollover
reference identity. Regression fixtures cover each trading-critical correction.

Validation: 422 Python tests passed (five existing dependency/research warnings); full Ruff
format/lint; mypy 108 files; Playwright desktop/mobile/focus/filter/scroll checks; isolated locked
server-only offline installation smoke. Labelled fixture screenshots are in docs/futures-screenshots.

No server cutover, restart, arming or test order. All six default mappings remain blocked for
explicit real product/expiry/strike-tolerance authority; approved mappings and cutover/arming
remain separate user-reviewed steps.
