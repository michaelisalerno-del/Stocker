# M1C Group O Late Revision V1

This immutable recovery contract addresses an operational source-publication
failure. On 2026-07-31, EODHD returned HTTP 200 with zero exact-session option
rows for all 20 frozen symbols. The existing recorder finalized that response
as the 2026-08-03 `missing_exact_chain` package.

The recovery does not replace that package. It preserves the original bytes,
uses a new append-only acquisition-attempt directory, and may append one
self-binding, hash-linked revision only if the exact Friday source becomes
available before the exact Monday XNYS open.

The V1 append-only mechanism is frozen generically for an exact D-1 source
publication correction. This deployment is separately bound to Friday
2026-07-31 for Monday 2026-08-03. Its one-off recovery command writes a
hash-bound start receipt before contacting EODHD, and recorder startup blocks
before construction of the IBKR adapter until the linked revision evidence is
complete.

The deployment freeze signs the exact failed-base SHA-256. Each empty-chain
attempt signs a `retry_after_utc` exactly 15 minutes after completion, and the
pre-adapter command waits and retries automatically on that persisted cadence.
Publication rechecks the XNYS-open cutoff immediately before the immutable
hard link. If the process stops after linking the revision but before writing
the completion receipt, restart verifies the signed start, exact candidate,
base hash, and revision, then appends only the missing signed completion
receipt. The dedicated `recovery_completion_receipt.json` is self-binding and
hash-links the deployment, start, failed base, staged candidate, published
revision, and acquisition-attempt receipt. Recorder startup verifies that
entire chain before constructing the IBKR adapter.

The pre-adapter integrity gate does not expire after the signal session. It is
permanent for this recorder version so a later restart cannot silently bypass
the recovery evidence. Decommissioning the gate requires a new signed recorder
version and deployment receipt; elapsed time alone is never authority.

The source observation is causal D-1 context for a future signal session. It is
not a Monday outcome and cannot change M1C formulas, Opening Leader identity,
rank, direction, recording frequency, option selection, or any order setting.
The runtime remains record-only with all broker-order capabilities disabled.

Runtime evidence locations:

- Failed base: `daily-context/group-o/2026-08-03.json`
- Attempts: `daily-context/source-cache/eodhd-group-o/2026-07-31/attempts/`
- Start receipt: `<attempt>/recovery_start_receipt.json`
- Completion receipt: `<attempt>/recovery_completion_receipt.json`
- Revisions: `daily-context/group-o/revisions/2026-08-03/`

If the exact chain remains unavailable at the Monday open, recovery fails
closed and M1C remains ineligible for that session.
