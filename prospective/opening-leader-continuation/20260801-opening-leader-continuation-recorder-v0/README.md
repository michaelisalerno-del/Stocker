# Opening Leader Continuation Prospective Recorder V0

This immutable package freezes a record-only prospective test of one claim:

> The single strongest stock in the frozen cohort at the causal opening checkpoint may
> continue outperforming by the end of the regular session.

The retrospective classification is `session_close_persistence_only`. It is not converted
into a strategy. C6 is primary, C12 is secondary, and their support is never pooled. The
candidate is always rank 1 by open-to-checkpoint return, ties resolve alphabetically, and the
direction label is always `LONG`. M1C is attached as context only.

Runtime observations do not belong in this directory. They are appended to the existing
prospective SQLite evidence database in `opening_leader_evidence_v0`, whose update and delete
triggers enforce immutability. Late data and corrections are new linked rows.

The module captures bounded underlying and option market-data snapshots through the existing
IBKR market-data-only adapter. Option evidence may support offline P20, P30, and BPS20 shadow
diagnostics, but no option policy or recommendation is authorized.

E0 is the first valid quote strictly after both the causal rank inputs and the immutable signal
receipt exist. The final continuous quote is frozen only after the regular session has ended and
remains separate from the non-executable official-close reference. Missing and late observations
are never substituted or rewritten.

The deployment freeze receipt binds these manifests and implementation sources after tests,
lint, type checking, synthetic dry-run, and restart recovery checks pass. Runtime startup must
verify it before requesting any Opening Leader market data.

**RECORD ONLY — ORDERS DISABLED**

Until each checkpoint independently reaches 60 valid sessions, three calendar months, and 15
distinct selected stocks, every dashboard view must state `PROSPECTIVE SAMPLE INCOMPLETE`.

Configure `paths.opening_leader_continuation_v0_root` to this directory, set a prospective
start strictly after the deployment freeze timestamp, and use the integrated recorder command:

```bash
stocker-prospective recorder run \
  --config /etc/stocker/prospective.yaml \
  --release-directory /opt/stocker/current
```

Runtime rows are stored in the existing prospective SQLite database table
`opening_leader_evidence_v0`; raw market callbacks remain in the existing partitioned event
store. The CLI verifies the deployment receipt and every bound source hash before constructing
or connecting the IBKR adapter.
