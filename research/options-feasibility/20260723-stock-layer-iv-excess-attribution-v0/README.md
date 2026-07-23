# Stock-Layer Attribution and IV-Excess Tail Quick Screen V0

This retrospective, research-only screen attributes the previously supported Branch C
increment across four frozen stock-information layers:

1. daily stock context;
2. intraday compressed-transition context;
3. route competition and route-resolution state;
4. explicit stock/options mismatch.

It uses only the exact frozen joined panel from Daily Stock + Front-Options Context Quick
Screen V0.1. It makes no EODHD requests, uses no intraday option quotes, calculates no
option P&L, and cannot place orders.

Run from the repository root, supplying the local frozen V0.1 panel when it is not present
inside the checkout:

```bash
uv run python research/options-feasibility/20260723-stock-layer-iv-excess-attribution-v0/run_screen_v0.py \
  --frozen-panel /absolute/path/to/front_options_cross_market_panel.parquet
uv run python research/options-feasibility/20260723-stock-layer-iv-excess-attribution-v0/audit_screen_v0.py
```

To rebuild only the Markdown report from completed artifacts, without repeating any fit or
resampling:

```bash
uv run python research/options-feasibility/20260723-stock-layer-iv-excess-attribution-v0/run_screen_v0.py \
  --report-only
```

The binding output is `artifacts/primary/decision.json`. The screen is movement-feasibility
evidence only—not an option strategy, profitability result, directional edge, prospective
validation, trading recommendation, or deployable system.
