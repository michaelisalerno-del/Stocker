# Stocker V2 inbox recovery and drain-priority plan

## Scope

Recover the exact durable 50,000-callback backlog for
stocker-v2-shadow-20260812t171500z, prevent receipt discovery and downstream
work from starving callback projection, and start a fresh shadow run.

This phase changes only V2 shadow market-data ingestion scheduling and one
SQLite index. It does not change ideas, models, thresholds, subscriptions,
broker configuration, proposal semantics, risk, execution, paper, or live
behaviour. IBKR remains loopback-only and read-only, with no account, order,
fill, or position surface.

## Implementation

1. Add a partial callback-inbox index matching the existing unreceipted
   terminal-row predicate used by receipt discovery and bind that discovery
   query to the index so SQLite cannot prefer the broad lifecycle index.
2. Preserve the 256-row ordered projection and receipt path.
3. In the continuous recorder, when a full leased batch proves projection is
   behind, publish the heartbeat and defer snapshot, idea,
   dynamic-subscription, and shadow work only after each leased callback is
   receipted. Offline replay keeps its per-batch downstream work.
4. Keep the 50,000-row hard limit, callback identities, ordering, receipt
   chain, bounded memory, one-writer rule, and all protected evidence.

## Operational recovery

1. Preserve a checked pre-recovery backup and restore it to a new rehearsal
   path.
2. Bind a new exact recovery program to the fatal run, generation, sequence
   range, release, callback manifest, receipt frontier, backup hashes, and
   deterministic output digests.
3. Rehearse broker-free. Require complete callback projection or explicitly
   verified deterministic failure, chained receipt coverage, no outside
   mutation, quick_check success, and no foreign-key violations.
4. Apply the exact program to production only after rehearsal. Resolve the
   INBOX_FULL incident only after terminal proof, stop the old run, and retain
   an unresolved discontinuity gap.
5. Deploy the reviewed release, migrate to schema 15, verify the index and
   safety boundaries, and start a fresh run ID attended.

## Acceptance

- Receipt discovery uses callback_inbox_unreceipted_terminal_idx.
- A full projection batch is terminal and receipted before downstream work is
  deferred.
- Downstream work resumes on a non-full batch without scientific differences.
- The recovered callback range remains complete, ordered, identity-bound, and
  receipted.
- Backlog trends downward under observed ingress without raising its cap.
- Web stays query-only and the runtime exposes no order capability.

Before the new run admits its first callback, rollback may restore the checked
pre-migration database and release together. After new admission, preserve a
checked backup and roll forward; never restore away newly admitted evidence.
