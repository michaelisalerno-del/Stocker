# Independent audit — M1C Asymmetric Downside Residual V1

`passed`

- Assessment/stress rows: 417/525.
- Endpoint returns, previous-close IV thresholds, inclusive states, and the canonical strict M1C movement label were reconstructed from primitive bars and prior-close ATM IV without importing the experiment target helper.
- Persisted scaler, coefficients, intercept, OOF quantiles, scores, and actions reproduced within 1e-12.
- M1C probability/tail membership, fresh episode IDs, A1 outputs, Tail Phase, and movement-consumed fields exactly match the frozen Tail Phase source artifact.
- All recorded output hashes, 1,000-draw bootstrap cells, and 1,000-draw permutation cells passed.
- No joint-probability or contaminated feature column was emitted.
- No protected 2026 outcome or order path was accessed.
