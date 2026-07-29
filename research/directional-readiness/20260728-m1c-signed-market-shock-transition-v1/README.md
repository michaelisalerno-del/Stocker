# M1C Signed Market Shock Transition V1

This is one narrow, preregistered retrospective direction experiment over
fresh frozen-M1C high-movement episodes. It tests two separate mechanisms:

1. shock continuation among `AMPLIFYING` stocks; and
2. shock resistance/opposition among `RESISTING` stocks.

The mechanisms are never combined or selected after outcomes are viewed.
VTI is the sole canonical broad-market proxy because it is already used by
the repository's causal market-direction baseline. All market and stock
response fields end at the M1C signal timestamp, before next-bar-open entry.

The study does not retrain or alter M1C, A1, Tail Phase V1, the frozen cohort,
the checkpoint grid, fresh episodes, option selection, recorder priority, or
any execution path. It does not calculate option P&L.

## Run

```bash
rtk uv run python research/directional-readiness/20260728-m1c-signed-market-shock-transition-v1/run_experiment.py
```

The runner applies date predicates to every row-bearing source read and fails
closed if a materialised session reaches 2026.
