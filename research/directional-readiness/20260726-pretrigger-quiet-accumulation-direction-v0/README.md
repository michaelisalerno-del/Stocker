# Pre-Trigger Quiet Accumulation / Distribution Direction Screen V0

This retrospective candidate experiment asks whether one frozen, symmetric
bar-derived marker is directional **before** the completed bar that triggers
the validated M1 movement gate.

The binding marker uses exactly five completed five-minute bars ending at
`T-1`. The full M1 trigger bar `T` is excluded from every direction feature.
Prospective entry remains the open of the first completed five-minute bar
after the trigger, so the experiment also measures how much movement remains
after that realistic gate timing.

The screen uses underlying-stock returns, five-minute OHLC, the existing
causal activity proxy, the frozen VTI market proxy, causal session VWAP, and
the repository's audited signed-pressure snapshots. The activity fields are
not asserted to be confirmed exchange volume. The signed-pressure source is
available only at even completed-bar checkpoints. It is used only on the exact
bar where that snapshot is available; intervening bars remain missing. Values
are not interpolated, carried forward, or redefined. Consequently the full
five-bar persistent-pressure portion of the requested hypothesis fails closed
when no exact per-bar series exists.

This is retrospective candidate evidence only. It is not direct order-flow
research, an option P&L backtest, prospective validation, paper readiness,
live readiness, or a trading strategy.

Run:

```bash
python research/directional-readiness/20260726-pretrigger-quiet-accumulation-direction-v0/run_screen_v0.py
python research/directional-readiness/20260726-pretrigger-quiet-accumulation-direction-v0/audit_screen_v0.py
```

Primary artifacts are written beneath `artifacts/primary/`; the readable
report is also copied to `reports/report.md`.
