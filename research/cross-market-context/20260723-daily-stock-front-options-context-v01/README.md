# Daily Stock + Front-Options Context Quick Screen V0.1

This retrospective research screen runs three independent branches:

- daily stock context for clean registered-loop completion;
- front-options-only context for completion and IV-relative movement;
- one isolated, non-compact back-expiry schema preflight.

It uses exact previous-session options observations, calculates no option P&L, performs no
bulk back-expiry download, and cannot place or describe executable trades.

Run from the repository root:

```bash
uv run python research/cross-market-context/20260723-daily-stock-front-options-context-v01/run_screen_v01.py
uv run python research/cross-market-context/20260723-daily-stock-front-options-context-v01/back_expiry_preflight.py
uv run python research/cross-market-context/20260723-daily-stock-front-options-context-v01/audit_screen_v01.py
```

The preflight reads `EODHD_API_TOKEN` only from the process environment, makes at most one
request with at most 100 records, and writes raw provider content only under ignored
`data/vendor/eodhd/options/` storage.
