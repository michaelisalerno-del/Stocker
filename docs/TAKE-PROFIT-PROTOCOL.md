# A resting take-profit from the moment of entry — frozen protocol (2026-10-06 ~12:15 UTC, on the user's "Freeze it", before the data it is judged on)

Many of the app's options trade above their entry price during the hour and give it back before the clock exit (52 of
84 traded clocks to 5 October). Report 4 of the user's research (2026-10-06) proposes selling into those moves with a
sell order that is already resting when the spike comes, so the position is sold as the liquidity provider instead of
hitting the bid afterwards. The frozen tests already cover a crossing take-profit (X3 of `L2-VETO-EXIT-PROTOCOL.md`), a
ratchet (the armed trail of `EXIT-SET-3-PROTOCOL.md`), a futures trail (E1 of `EXIT-SET-2-PROTOCOL.md`) and a resting
sell from minute 55 (scheme D of `EXECUTION-SET-2-PROTOCOL.md`). This one adds a resting sell for the whole hour. It is a
paper comparison on the hourly ledger; nothing changes the runtime and no order is sent. The hash of this file is in
`TAKE-PROFIT-PROTOCOL.sha256`; any later change is an amendment, dated and stated as such, made before the results are
looked at.

## Data
`tp.csv`, written by `tp.py` in the hourly ledger (sha256
e09b684e8d08679fc60acbb83b1ae159f9944b29a666a470ba907e10c5fb8eb8 at freezing) after `build.py` (sha256
69addc28dd9f7c3d97f367b2a4509f3c371f2965979eb9a5d8deae35bdd813f2). It imports the quote, tick and resting-order functions
of `execution2.py` (sha256 f0e42de4bf3e54966c81d5145bc9d2449a53b1e00cfb3d29cfdff683a0cf1781, frozen with
EXECUTION-SET-2) unchanged. The module docstring of `tp.py` is the exact specification.

## Baseline
As in EXECUTION-SET-2: buy at the ask of the last quote at or before t0 = clock + 3 s; sell at the bid of the last quote at
or before e0 = clock + 60 min + 3 s (both at most 30 s old; no bid = 0), or at the intrinsic value when the option settles
within a minute of e0.

## Rule TP (no free parameter; the defaults of Report 4)
- At t0 + 3 s (after the entry fills), rest a sell at L0 = the entry ask × 2 (+100%), and at least k ticks above the ask
  (ES 8, NQ 6 on its 0.05 grid or 3 on its 0.25 grid, CL 3, GC 3), rounded up to the option's tick grid at that price.
- From clock + 40 min, move it to L1 = the ask × 1.5 (+50%), with the same minimum.
- Fill: a quote received at u shows the bid at or above the price live at u − 3 s, or a new trade prints strictly above
  it (a trade at the price does not count), with u from placement + 3 s to e0, or to a minute before expiry for an option
  that settles first. Otherwise sell as the baseline.
- `tp_saving` = (TP's exit − the baseline's exit) / the entry ask, per clock.

## Verdict (once, after 2026-11-27)
- Judged clocks: from the first clock after the freezing commit (the 09:00 New York clock of 2026-10-06) to 2026-11-27
  16:00 New York, every market and session; halves split at 2026-10-31.
- TP PASSES only if the mean `tp_saving` is positive with t > 2.5 across days (the daily mean), on at least 100 scored
  clocks on at least 15 days, and is positive in both halves. It is judged per market as well, at the same bar on at least
  40 clocks on at least 10 days in that market, because Report 4 expects the gain where ticks are coarse (CL, GC).
- Tail check, required for a pass: the summed baseline return of the clocks in the baseline's top 5% must fall by less
  under TP than TP gains on the other 95%. Profit-taking that pays only by cutting the rare large winners does not pass.
- Reported regardless (not judged): `tp_bid_only_saving` (filled only when the bid reaches the price) and
  `tp_maker_saving` (also filled when the ask sits at or below the price on 3 consecutive quotes with volume rising, half
  of such fills kept by a seeded coin; an optimistic maker estimate); the fill rate and minute; the durable MFE at the bid
  (best bid held for 3 s) against the hour's end, and the share of clocks touching +25/50/100/200% with their mean final
  return (E[final | touched +x] − x, the continuation value). The sign-randomised null of Report 4 (§1.4) is part of the
  post-27 November analysis, not of this verdict.
- Paper caveat, as in the other execution tests: a real resting order joins the queue at its price, and whether Saxo
  supports a related take-profit order on FuturesOption is not yet probed. A pass earns a one-lot live test, not a change
  to the app.

## Expectation stated now
On a fair-game option path a target leaves the mean almost unchanged and only reshapes the distribution (Report 4,
§3.4). Under the strict rule TP is close to X3 (+100%), which crosses at the bid when it reaches the same level; TP's
difference is the trade-print evidence, the tick minimum and the step-down at minute 40. A pass therefore needs either
reversion after the touch or a real maker gain.

## What was seen before this was written
Counts only: 60 recorded clocks score, 11 of them reach the target under the resting rule. No return was computed before
freezing. The look on those clocks (1–6 October, not judged) is reported after the freezing commit.
