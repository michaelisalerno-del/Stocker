# Daily Stock × Options Regime Context Quick Screen V0

This is a retrospective, research-only quick screen joining the frozen dense clean-advance
intraday panel to causal daily stock context and exact previous-session EODHD options context.
It tests forecast information and underlying movement relative to prior-close IV; it does not
calculate option P&L, search option strategies, access a broker, or modify production runtime.

Run from the repository root:

```bash
uv run python research/cross-market-context/20260723-daily-stock-options-regime-context-v0/run_screen_v0.py
uv run python research/cross-market-context/20260723-daily-stock-options-regime-context-v0/audit_screen_v0.py
```

The runner discovers the repaired V0.1 exact-date canonical cache from its frozen source
manifest. `download_gap.py` is a cache-first bounded gap inspector; it never prints credentials.
All material outputs are written to `artifacts/primary/`, with the human report mirrored under
`reports/`.
