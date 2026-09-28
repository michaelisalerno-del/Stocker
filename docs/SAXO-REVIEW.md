# Review of the Saxo migration

Reviewed against the existing application baseline `8f22d2b6e052d63ec440410c7ecb8e2f66b432d8`
and the user's Saxo-only migration task. Independent reviewers assessed implementation commit
`c74e68a`; targeted re-reviews checked the subsequent fixes. Reviews and fixtures are offline.

## Standards

Three findings; highest initial severity P1; all resolved:

- **P1 — Broker-fill crash boundary.** A committed fill followed by an interrupted order-counter
  update could make replay calculate an invalid incremental price. Fill, counters and audit now
  commit atomically. Replay derives the existing amount/cash from durable fills and can repair the
  earlier boundary. A crash-injection regression verifies rollback and idempotent recovery.
- **P2 — Stale internal FX sizing.** Internal fill admission could use an older conversion from
  the plan. It now validates current FX and recalculates whole-contract premium and fees against
  the £10 ceiling immediately before filling.
- **P2 — Shared size freshness.** A bid-size update could revive an old ask size. Side-specific
  timestamps now govern both freshness and consumed displayed size.

Targeted re-review: no remaining concrete defects in these fixes. The claimed duplicate EXIT
finding was withdrawn because the existing durable pending-order guard already prevents it.

## Spec

Three findings (one shared with Standards); highest initial severity P1; all resolved:

- **P1 — Gapped subscriptions did not rebuild.** A session downgrade or disabled subscription
  now closes the stream and obtains fresh subscription snapshots with bounded reconnect backoff.
  It never upgrades the Saxo session automatically or displays the previous book as current.
- **P1 — Shared size freshness.** Resolved by the side-specific timestamps described above.
- **P2 — Buffered receipt timestamps.** Updates arriving before a subscription's REST snapshot
  retain their original UTC receipt time when replayed, so waiting for the snapshot cannot make
  stale quotes appear fresh. The archive retains the same original receipt.

Targeted re-review: no remaining concrete defects in these fixes. Tests explicitly verify the
gap/reconnect signal, original timestamps and stale side sizes. The duplicate EXIT finding was
also withdrawn after checking the existing guard.

Final validation: 426 Python tests, Ruff formatting/lint, Mypy, locked browser checks and isolated
server-only smoke pass. Additional admission tests reject indicative option quotes, unverified
current sessions and missing trading permission. Account entitlements, actual Saxo payloads,
recording rights, broker-paper execution and production cutover remain unverified.
