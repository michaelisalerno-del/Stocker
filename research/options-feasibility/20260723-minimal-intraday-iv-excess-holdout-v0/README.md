# Minimal Intraday Stock → IV-Excess Holdout Validation V0

This is a strictly frozen, retrospective validation of one binding hypothesis:
previous-close front-options context plus the current intraday H0 stock condition
identifies a frozen 2024-defined top-5% probability tail whose subsequent
15-minute absolute underlying movement exceeds the previous-close ATM-IV
expectation.

The model is fitted only on 2024. The period 2025-01-01 through 2025-08-22 is
reconstructed only for predecessor compatibility. The only newly opened holdout
is actual XNYS sessions from 2025-09-01 through 2025-12-31. Observations dated
2026-01-01 or later remain protected.

The experiment contains exactly two primary weighted logistic models:

- `M0`: frozen front-options Group O plus stock fixed effects.
- `M1`: Group O plus frozen intraday-H0 Group I plus stock fixed effects.

Daily stock features and regimes, route-competition features, route-state
controls, and hand-built stock/options mismatch features are explicitly
excluded. Options are exact previous-session EOD observations; same-day, future,
or older forward-filled chains are rejected. No option P&L, intraday option
quote, direction, execution, broker, or deployment surface is opened.

Run the bounded workflow from the repository root:

```bash
python research/options-feasibility/20260723-minimal-intraday-iv-excess-holdout-v0/download_holdout_options.py
python research/options-feasibility/20260723-minimal-intraday-iv-excess-holdout-v0/run_screen_v0.py
python research/options-feasibility/20260723-minimal-intraday-iv-excess-holdout-v0/audit_screen_v0.py
```

Provider responses and canonical option records remain under ignored
`data/vendor/eodhd/options/` cache paths. Only small derived artifacts are
eligible for version control.

Observed run result: `blocked_quick_resource_limit`. The exact-date acquisition
reached the frozen 350,000-record ceiling after 1,450 of 1,700 complete
stock-session requests. The partial cache was not modeled and no binding
holdout outcome was opened.
