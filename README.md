# SLRNO

A Saxo **paper record**: it watches three rules every New York trading day and records what each
would have made, at Saxo's own prices. It never places an order. The rules are frozen in
[the paper-record protocol](docs/PAPER-RECORD-PROTOCOL.md):

1. **First candle on CFDs.** At 09:35 New York the 09:30-09:35 five-minute candle of the US Tech 100
   and US 500 CFDs is compared with EMA(12) of the CFD's five-minute closes (overnight included):
   above -> long, below -> short. Each paper trade is sized so its stop (100 bp Nasdaq, 50 bp S&P)
   costs $200, and is held to the 16:00 close.
2. **The same Nasdaq signal as a QQQ option.** The same-day call (long) or put (short) with delta
   nearest 0.40, bought at the ask at 09:35 and sold at the bid at 13:00.
3. **Connors on QQQ.** At 15:50, close above its 200-day average and RSI2(today) + RSI2(yesterday)
   below 35 -> the next expiry's call with delta nearest 0.50, sold at the bid at 15:55 on its
   expiry day.

```sh
uv sync --locked --no-default-groups
uv run --no-sync stocker paper-run --config configs/paper.example.yaml --database .stocker/paper-record.sqlite3
```

The Saxo client allows a short list of GET endpoints and nothing else (`saxo_client.py`). QQQ prices
reach the app 15 minutes late whenever SaxoTraderGO holds Saxo's single real-time slot; each QQQ step
is then read 15 minutes after its time and recorded as of that time. CFD prices are always live.

The dashboard has three pages: **Today** (signals, the day's events and paper trades), **Record**
(every paper trade and totals per method) and **System** (Saxo login and data state).

Until 15 October 2026 this repository was a futures-options paper runtime (ES, NQ, CL, GC). That code
was removed when the user cancelled the CME, COMEX and NYMEX data
([note](docs/FUTURES-PAPER-TESTS-ENDED-20261015.md)); it remains in git history, and the documents in
`docs/` that describe it are kept as the record of that work.

[Deployment](docs/DEPLOYMENT.md) · [Setup and OAuth](docs/SAXO-SETUP.md) ·
[Current server evidence](docs/CURRENT-DEPLOYMENT.md)

Run `bash scripts/check.sh` for format, lint, types, Python tests, browser checks and server-only smoke.
