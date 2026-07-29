# M1C Prospective Opening Reversal V1

This is a narrowly preregistered, research-only shadow experiment for
checkpoint-6 fresh high-M1C `FIRST_ENTRY` episodes after a severe signed VTI
opening transition.

The frozen action is:

- negative severe VTI opening transition → `CALL`;
- positive severe VTI opening transition → `PUT`;
- every other or incomplete state → `ABSTAIN`.

Equivalently, `prediction_sign_v1 = -opening_transition_sign_v1`.

The implementation preserves the always-on VTI and frozen 20-stock five-minute
bar universe, promotes at most one eligible stock for higher-cost recording,
and requires only one 1DTE nearest-ATM call/put pair. Twelve market-data lines
remain reserved. Optional comparisons cannot borrow from that reserve or
invalidate an underlying-direction episode.

The first 20 valid sessions are engineering-transfer only. No pre-activation
2026 outcome may be backfilled or analysed. No order route or order method is
part of this experiment.

The one-time activation package is created with:

```bash
rtk uv run python research/prospective/20260729-m1c-prospective-opening-reversal-v1/build_activation_artifacts.py --activate
```

After activation, the builder refuses to overwrite the immutable receipt.
Verification is read-only:

```bash
rtk uv run python research/prospective/20260729-m1c-prospective-opening-reversal-v1/build_activation_artifacts.py --verify
```
