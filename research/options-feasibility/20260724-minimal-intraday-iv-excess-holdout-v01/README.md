# Minimal Intraday Stock → IV-Excess Holdout Validation V0.1

This is the strict, frozen continuation of V0 after its
`blocked_quick_resource_limit` stop. It reuses every verified complete V0
receipt, repairs the one interrupted logical request, downloads only missing
requests, and opens holdout movement outcomes only after options coverage,
historical reconstruction, model coefficients, and weighted 2024 thresholds
are frozen.

The experiment is retrospective and research-only. It evaluates underlying
movement relative to previous-close option-implied expectation. It does not
calculate option P&L, use intraday option quotes, access a broker, enable
execution, or define a deployable strategy.

Run only the scoped workflow:

```bash
uv run python research/options-feasibility/20260724-minimal-intraday-iv-excess-holdout-v01/download_holdout_options.py
uv run python research/options-feasibility/20260724-minimal-intraday-iv-excess-holdout-v01/run_screen_v01.py
uv run python research/options-feasibility/20260724-minimal-intraday-iv-excess-holdout-v01/audit_screen_v01.py
```

Raw and canonical EODHD data remain under ignored cache directories. No API
credential belongs in this directory or in Git.
