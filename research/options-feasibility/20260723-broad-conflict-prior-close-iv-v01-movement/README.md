# Three-stock prior-close IV movement outcomes V0.1

This no-network amendment applies the frozen underlying-movement definitions to the exact cached
AAL, MSTR, and WULF option-pair probe. It uses the clean V0.2 structural population and frozen
five-minute trace for three signal dates.

All clean sampled rows receive underlying 10-, 15-, 30-, and 60-minute descriptive outcomes.
IV-relative outcomes require a valid exact-previous-session ATM pair; the failed 2024-01-16 WULF
pair remains unavailable and is not replaced.

The sample is intentionally below the full experiment's coverage and stability gates. No model,
bootstrap, null refit, matched control, intraday option fill, option P&L, executable return,
strategy result, prospective validation, or trading-utility claim is produced.

Run with:

```bash
.venv/bin/python \
  research/options-feasibility/20260723-broad-conflict-prior-close-iv-v01-movement/run_movement_outcomes.py
```
