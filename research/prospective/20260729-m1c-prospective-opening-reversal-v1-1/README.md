# M1C Prospective Opening Reversal V1.1

V1.1 is a timing-only operational addendum to the immutable M1C Prospective
Opening Reversal V1 preregistration.

The scientific rule is unchanged:

- negative severe VTI opening transition → `CALL`;
- positive severe VTI opening transition → `PUT`;
- every other or incomplete state → `ABSTAIN`.

Equivalently, `prediction_sign_v1 = -opening_transition_sign_v1`.

The V1 requirement to complete a prediction receipt strictly before 10:00 was
not physically satisfiable because the sixth frozen predictor bar runs from
09:55 through 10:00. V1.1 permits the receipt just after that boundary only
behind a causal barrier:

1. Complete the sixth predictor bar.
2. Score all 20 frozen stocks.
3. Persist all 20 immutable prediction receipts.
4. Persist the session's causal-barrier audit.
5. Release buffered entry/post-entry data to decision and outcome surfaces.

Raw append-only archival may continue while the barrier is closed. If the
barrier cannot close, the scientific session fails closed while the core M1C
recorder continues. The nominal 10:00 entry remains non-actionable.

V1.1 preserves the exact V1 scientific rule, thresholds, cohort, checkpoint,
freshness, Tail Phase, A1, 15-minute horizon, option policy, 12-line reserve,
degradation order, and no-order boundary. It requires a fresh recorder run and
restarts the 20-session engineering-transfer phase.

Activate once:

```bash
rtk uv run python research/prospective/20260729-m1c-prospective-opening-reversal-v1-1/build_activation_artifacts.py --activate
```

Verify read-only:

```bash
rtk uv run python research/prospective/20260729-m1c-prospective-opening-reversal-v1-1/build_activation_artifacts.py --verify
```

The builder reads only frozen V1 metadata and empty artifact schemas. It does
not open protected historical outcomes, connect to IBKR, enable order routing,
or place an order.
