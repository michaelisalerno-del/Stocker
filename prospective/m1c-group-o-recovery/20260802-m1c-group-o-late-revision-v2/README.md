# M1C Group O Late Revision V2

V1 failed closed and remains immutable. Its first real-source attempt received
Friday 2026-07-31 EODHD option rows but admitted none: 5,965 rows carried a
provider `dte` one day below the calendar interval encoded by the EOD resource
identity, and 399 rows had mismatched bid/ask observation dates. It wrote no
candidate package, no revision, and did not alter the failed base.

V2 is a new source-normalization freeze. It does not rewrite or reuse V1
attempt `0001`. It downloads a fresh attempt after verifying the exact V1
failure receipt. The provider documents `dte` as “days until expiration” and
documents the option EOD resource identity, expiration, bid date, and ask date
separately. V2 therefore freezes this deterministic rule:

```text
dte = expiration_date - exact_eod_resource_observation_date
```

Provider `dte` remains source evidence but cannot admit or reject a row,
including when its value is malformed, fractional, negative, or inconsistent.
Every provider value and its classification are written to an append-only
`provider_dte_diagnostics.json` file whose exact bytes are hash-linked from the
signed acquisition-attempt receipt. The immutable raw responses remain the
source of the original value. The exact EOD resource date must still be Friday
2026-07-31. Bid and ask timestamps must both map to that date in New York;
mismatches remain rejected. Underlying symbol, expiration, strike, option type,
finite numeric fields other than provider `dte`, duplicate resolution, and all
other frozen checks remain unchanged.

The official field descriptions are published in the
[EODHD US Stock Options Data API documentation](https://eodhd.com/marketplace/unicornbay/options).
This source adapter change cannot alter M1C formulas or Opening Leader rank,
identity, direction, frequency, option selection, or any order setting.

Before a request, V2 verifies the signed deployment, original failed base,
exact canonical cohort, V1 deployment receipt, exact V1 start receipt, and V1
failed attempt. The chronology must be V1 deployment freeze, V1 start, V1
attempt completion, V2 deployment freeze, then V2 start. It then writes a V2
start receipt and allocates a new attempt; reconciliation verifies and skips
the immutable V1 `0001` directory before reading V2 attempts. Successful
publication remains a self-binding, hash-linked append-only revision strictly
before the Monday XNYS open. A V2 completion receipt links the deployment,
start, base, new candidate, revision, and acquisition receipt. Recorder startup
verifies the entire chain before constructing the IBKR adapter, permanently for
this recorder version.

Runtime evidence locations:

- Failed base: `daily-context/group-o/2026-08-03.json`
- Failed V1 attempt: `daily-context/source-cache/eodhd-group-o/2026-07-31/attempts/0001/`
- Fresh V2 attempts: subsequent numbered directories under the same `attempts/`
- Revisions: `daily-context/group-o/revisions/2026-08-03/`

The runtime is record-only. It has no broker-order construction, preview,
simulation, staging, or placement surface.
