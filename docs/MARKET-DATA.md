# Saxo stream and recorder policy

One server data owner per environment/root keeps all configured five underlyings subscribed even
with no browser. Browsers read existing authenticated APIs and never subscribe to Saxo.
Price/depth subscriptions request 1,000 ms; the option discovery board requests its supported
2,000 ms minimum. Actual returned RefreshRate is reported. The selected option receives a regular
price subscription for its account and intended quantity of one. Board quotes are not executable prices.

The bounded binary parser handles initial snapshots, field patches, null clears, price-array
replacement, indexed option-board updates, heartbeats and duplicate opaque message IDs. A separate
local ingest sequence is retained. Conflicting replays, reset/disabled subscriptions, session downgrade,
disconnect or token reauthorization invalidate reconstructed books. Fresh snapshots are required.
A heartbeat or depth update cannot freshen old bid/ask prices. Quote freshness is conservative at
five seconds; stale L2 ladders clear. Diagnostic depth never changes entry or exit decisions.
No sampled update is described as an exchange tick, cancellation or aggressor trade.

The recorder retains at most the latest 900 seconds plus the **reconstruction checkpoint immediately
before the first retained update**. Expiring records advance that checkpoint. Local UTC receipt,
provider envelope/timestamps where supplied, environment, actual identity, local sequence, provider
ID, gaps and duplicates are recorded. Depth and L1 are the actual delivered fields. Recorder restart
cannot reconstruct pre-start L2; bars never masquerade as recovered raw depth.

Each new frozen clock occurrence has a stable environment/contract/strategy/time ID in SQLite.
Duplicate refreshes/quotes do not trigger captures. Veto, capacity, budget, missing-option and
monitor-only skips are recorded. Only the affected future and its relevant subscribed options are
pinned. Options subscribed after the event are attached with their actual shorter history.
Pre-event available coverage is explicit. Capture lasts at least 60 minutes after each event, through
any linked open trade, then another five minutes after verified closure. Overlapping windows share
one data segment and retain distinct event identities; a completed overlap bridges intervening
buffer rows by local sequence instead of duplicating data.

Appendable gzip members contain compact JSONL. A durable manifest intent precedes raw writes;
chunks fsync, then manifests replace atomically. A restart truncates only an incomplete gzip tail,
preserves prior complete members and labels interrupted captures INCOMPLETE. Existing raw evidence
is not overwritten. An interrupted capture is not advertised as continuous after restart. Required
paper position management recovers separately from its durable execution ledger.

Defaults, selected against this workstation's 230 GiB available disk (server resources unverified):

| Resource | Limit/default |
|---|---:|
| Rolling window | 15 minutes, 32 MiB aggregate, maximum 32 instrument windows |
| One message | 256 KiB |
| Raw write queue | 512 items and 8 MiB |
| Concurrent/recent captures | 32; 256 event identities per shared interval |
| Event archives | 2 GiB; 64 KiB reserved for bounded failure metadata |
| Minimum free disk | 2 GiB |
| Completed one-minute bar cache | 128 MiB, separate compressed contract/day files |
| Runtime logs | 2 MiB plus 3 rotated files |
| REST | 32 queued calls; 4 MiB response; conservative 0.55 s spacing |
| Subscriptions | 32 total, at most 16 regular option quotes |
| Pending snapshot updates / WebSocket queue | 1 MiB / 16 bounded frames |

Time expiry and byte limits both apply. Limits shorten reported history rather than pretending 15
minutes exist. Discretionary option subscriptions expire after 15 minutes unused, unless owned or
referenced by an active capture. Raw quotes are never inserted into the execution database.
Completed chart samples are compactly retained only with recording permission; they include provider
identity and DataVersion. No continuous one-second summaries or permanent session raw files exist.

At quota, reserve, count or queue limits, raw capture stops with an explicit incomplete reason and
STORAGE_LIMIT/UNAVAILABLE status. The RAM stream and independent position manager continue. No saved
event evidence is automatically deleted. Disk I/O/compression run in a worker thread. Stale temporary
files are removed on startup. This does not introduce Kafka, Redis or another service stack.

Download `/api/recordings/{segment}` and `/api/recordings/{segment}/manifest` under existing dashboard
authentication. `POST /api/recordings/{segment}/prune` is explicit and only permits completed,
unreferenced, unprotected segments outside active/overlap windows. Any execution-ledger reference
blocks pruning. Download/backup never automatically unlinks evidence. Move protected research to your
approved archive process with manifest/hash verification; do not delete active data to clear a warning.

Official schemas and semantics checked on 2026-09-28:
[pricing](https://www.developer.saxo/openapi/learn/pricing),
[WebSocket streaming](https://www.developer.saxo/openapi/learn/plain-websocket-streaming),
[option board](https://www.developer.saxo/openapi/learn/options-chain),
[reference data](https://www.developer.saxo/openapi/learn/reference-data),
[charts](https://www.developer.saxo/openapi/learn/chart),
[rate limits](https://www.developer.saxo/openapi/learn/rate-limiting),
[reference schemas](https://www.developer.saxo/openapi/referencedocs).
These documentation checks do not establish this account's entitlements or recording permission.
