# EMA12 first candle — amendment 1: gamma state at 09:35, reported only (2026-10-13, on the user's "Yes", before the sample starts)

Adds a reported split to `EMA12-FIRST-CANDLE-PROTOCOL.md` (sha256 af5c381b...). The verdicts, their thresholds, the judged sample,
the judging point and `emafc.py` (sha256 b8cd1369a4f1268be14f4f5bee3835fdd8b942a5d7202d6f869944ba13a89eb8) are unchanged.
The hash of this file is in `EMA12-FIRST-CANDLE-AMENDMENT-1.sha256`.

## Why
`~/Codex/2026-10-13-open-greeks/` (OG_PREREG.md sha 99131b84, written before running): the opening implied-vol move adds nothing
(registered tests failed), but in the reported part the EMA12 signal did better when the 09:35 price sat BELOW the open-interest
gamma flip (dealers short gamma): +8.1 bp/day (t 2.35, |move| 75 bp) vs +2.5 bp above (46 bp), SPY/QQQ November 2023 – October 2026,
same order in SPY, QQQ and both halves but each difference t 0.6–0.9. A lead only; this amendment records it forward.

## What is recorded
`emafc_gamma.py` (hourly ledger folder) reads `emafc.csv` and the chain board sampled at 09:35 (`build.chain_at`). The board carries the
same-day expiry only (about 61 strikes: NQ about ±0.8%, ES about ±2% of the price), so the flip of the research (all expiries, ±5%)
cannot be rebuilt. Recorded instead, per market and day: `gex_net_0935` = sum over the board of OI × Black-76 gamma (each strike's
mid volatility, time to the expiry instant) × F², calls +, puts −; `gamma_state` LONG if > 0, SHORT if < 0; `flip_0935` = the zero
of that sum repriced over F × [0.97, 1.03] nearest F, only when it lies inside the board's strikes. Output `emafc_gamma.csv`.

## Reported (not judged), at the protocol's cost judging point
NQ and ES: futures outcome (all and strong) and option carry by `gamma_state`. Before that point the script prints counts only.
A difference here is evidence for a later protocol, not a verdict.

## Script
- `emafc_gamma.py` sha256 c97a4ddcc04a88b0bf1c39bf7792f957596c93cff29073d477aed7865a0dde1a (uses `build.segments`, `build.chain_at`,
  and `emafc.START`, `EARLIEST`, `MIN_OPT_DAYS` unchanged). Run by hand after `emafc.py`.

## Disclosure
Run on the look days only while writing it. States seen: ES LONG 6 and 9 Oct, SHORT 7 and 8 Oct; NQ the same; no board at 09:35 on
1, 2 and 12 Oct. Outcomes of those days are listed in the protocol's disclosure. Nothing from 14 October on has been computed.
