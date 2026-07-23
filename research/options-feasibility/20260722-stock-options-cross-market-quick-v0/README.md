# Stock ↔ Options Cross-Market Information Quick Screen V0

This retrospective research-only screen keeps two questions separate:

1. Does the exact previous trading session's options state improve the frozen
   clean two-to-three-bar registered-loop completion forecast?
2. Do frozen compressed-transition and route-competition features improve the
   forecast that subsequent 15-minute underlying movement exceeds the previous
   close's option-implied expected absolute movement?

Run the bounded screen:

```bash
uv run python research/options-feasibility/20260722-stock-options-cross-market-quick-v0/run_screen_v0.py
```

Run the independent lightweight audit:

```bash
uv run python research/options-feasibility/20260722-stock-options-cross-market-quick-v0/audit_screen_v0.py
```

The runner uses only the committed frozen structural artifacts and the existing
untracked EODHD options cache. It never downloads options data. If cached support
misses any quick support gate, it writes the exact stock/date/month coverage and
request gap, emits `blocked_insufficient_cached_options_coverage`, and performs no
model, bootstrap, or null refits.

No output is options P&L, an intraday option fill, an executable return, economic
edge, prospective validation, trading utility, or a deployable strategy.
