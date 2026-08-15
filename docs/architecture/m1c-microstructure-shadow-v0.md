# M1C microstructure shadow V0

This is a collection-only sidecar to the frozen prospective M1C recorder. It does not
change M1C, trigger an episode, score a direction, or expose broker mutation.

## Reused transport

The recorder already starts one protected `reqMktData` Level-I stream for every member
of `anchor_frozen_20` before episode processing. It also starts the existing VTI
Level-I stream and the existing five-minute bar streams. The shadow collector copies
the 20 stock Level-I callbacks after durable normalization; it requests zero additional
market-data lines. Tick-by-tick BidAsk/Last and depth remain selective promotion-only
feeds. If an already-selected Last stream exists, its raw trades may be copied as
optional evidence. Its active/inactive lifecycle is a separate raw event, while each BBO
row and session summary state whether trades were available. Optional stream loss never
blocks BBO capture.

## Level-I event and ordering contract

IBKR's `tickPrice` and `tickSize` callbacks carry no exchange/provider event timestamp
and no exchange sequence number. Stocker therefore leaves `provider_timestamp_utc`
null for Level-I BBO rows. At the official callback boundary it records a UTC receive
timestamp, a process-monotonic nanosecond value, a durable inbox `source_sequence`, and
the connection generation. `source_sequence` is the authoritative callback-arrival
order within a run. Parquet ordering is provider timestamp when one exists, then local
monotonic receive order, source sequence, and stable event identity. Rows are never
deduplicated by timestamp, so legitimate same-timestamp callbacks remain distinct.

Tick-by-tick callbacks contain IBKR's whole-second provider timestamp. Stocker retains
that value, but IBKR does not expose an exchange sequence for these callbacks either.

Every emitted bid, ask, bid-size, or ask-size callback is retained, including partial,
locked, crossed, and invalid quote states. Last price, last size, and volume Level-I
callbacks are not copied into the shadow BBO event type. No quote is synthesized,
interpolated, or forward-filled.

## Dataset and continuity

Runtime activation writes:

- `shadow_collection_manifest.json`
- `shadow_research_contract_v0.json`
- `universe_manifest.json` with exact qualified IBKR conIds
- `collection_quality_summary.csv`

under a collision-safe activation namespace:

`<shadow_microstructure_root>/m1c_microstructure_shadow_v0/{pipeline_pilot|confirmatory}/activation_id=.../`

Pilot and confirmatory artifacts therefore cannot overwrite or mix with one another,
and each recorder run gets an immutable run identity file. Raw Parquet follows the
existing immutable layout beneath that namespace:

`data_source=ibkr/session_date=YYYY-MM-DD/symbol=SYMBOL/event_type=.../hour=HH/`

Subscription start, connection loss, restoration, subscription rebuild, and permission
failures are separate raw control/gap events. Per-connection quote state is cleared on
data loss, so a gap is never hidden by a synthetic or pre-gap unchanged quote. Pilot rows
carry `pipeline_pilot=true` and `confirmatory_eligible=false`.

## Activation boundary

Shadow mode is disabled by default. A pilot activation requires an explicit timezone-
aware activation timestamp and research-contract path. A pilot is bounded to the first
two complete XNYS sessions after activation and all of its rows are ineligible for the
confirmatory sample. Confirmatory activation also
requires a readiness report whose classification is exactly
`READY_FOR_SHADOW_COLLECTION`. The collector then freezes the first 20 complete XNYS
sessions whose market opens are at or after activation; it never extends that list from
observed results. A confirmatory start re-hashes the exact pilot manifest and its
all-symbol subscription/reconnect evidence, and rejects a READY document copied from or
edited for another pilot.
