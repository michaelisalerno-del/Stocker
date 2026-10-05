# Pressure–Progress: direction, entry and exit from the futures book and price progress — frozen unseen (2026-10-05)

The user brought a three-module research specification ("Pressure–Progress": direction, whether the option is worth
entering now, whether a trade has failed) whose parameters were set before any of the app's data was examined against
it, and chose to freeze it without a preliminary replay so that the test is clean. It is a paper experiment on the hourly
ledger: the app keeps its method unchanged, nothing below changes the runtime and no order is sent. The hash of this file
is in `PRESSURE-PROGRESS-PROTOCOL.sha256`; any later change is an amendment, dated and stated as such, made before the
results are looked at.

## Provenance
The rules, formulas and thresholds below are the user's specification as written (imbalance, progress, minimum
movement, persistence, the 60-second breakout, the 20-second limit, the cost-drag limit, the ratcheting boundary). Where it
left something undefined for computation, this document fixes it and marks it [definition added]. No result of these rules
on any recorded data had been computed when this was frozen. Earlier looks at the same recordings (1–5 October) found the
futures book (depth, imbalance) did not separate trades, and that buying after the future had already moved the option's
way did worse than buying at the clock; those are the reasons the rules might fail, not inputs to them.

## The underlying state, every 5 seconds
- Sampling [definition added]: at every whole 5 seconds from 60 s before a clock to the clock + 60 min, the futures book is
  the last CURRENT book-flow message of the clock's market at or before that time, if at most 5 s old ("healthy");
  otherwise the sample is unhealthy. Persistence counts elapsed healthy time; an unhealthy sample resets it.
- Book pressure: I = (Σ Bᵢ/i − Σ Aᵢ/i) / (Σ Bᵢ/i + Σ Aᵢ/i) over the nearest five bid and ask levels (Bᵢ, Aᵢ the displayed
  sizes at level i). The pressure is the median I over the healthy samples of the last 15 seconds. Bullish pressure:
  at least +0.20; bearish: at most −0.20.
- Midpoint: (best bid + best ask) / 2 of the futures' level-1 quote.
- Progress: E = net midpoint movement over the last 30 seconds / total absolute midpoint movement over those 30 seconds,
  from the 5-second samples; E = 0 if the midpoint did not move. Bullish progress: at least +0.40; bearish: at most −0.40.
- Minimum movement: the 30-second net midpoint movement must exceed max(2 ticks, 0.25 V) in the direction, where V is the
  median high–low range of the previous 15 completed one-minute futures bars.
- Direction state: BULLISH when bullish pressure, bullish progress and the minimum upward movement all hold continuously
  for 10 seconds of healthy samples; BEARISH likewise downwards; otherwise UNCLEAR.
- No-Level-2 control: the same state with the pressure condition removed (progress, minimum movement and persistence only).

## The three comparisons, each against the unchanged baseline
The baseline is the app's selected option bought at the ask at the clock and sold at the bid at the clock + 60 minutes
(the ledger's `opt_ret_ask_to_bid`), on every clock with the app's option, traded or skipped. A policy that does not buy
earns 0. Paper fills [definition added]: the option's recorded two-sided quotes, ask in and bid out, at most 30 s old.

1. Entry, confirmation mode (the baseline's side and contract): at the clock, freeze the futures' highest and lowest
   midpoint over the previous 60 seconds and V. The entry qualifies at the first 5-second sample within 20 seconds after
   the clock at which (a) the direction state agrees with the option's side (BULLISH for a call, BEARISH for a put), (b) the
   midpoint is at least one tick above that prior high (call) or below that prior low (put), (c) the future has not moved
   more than V in that direction since the clock [definition added: V is the chase limit], and (d) the option's immediate
   cost drag — (buy cost − immediate net sale proceeds) / buy cost, with the app's fill, commission and FX model — is at
   most 15%. Buy at the ask then; sell at the original exit (clock + 60 min). No qualifying sample: no trade.
2. Exit only (the baseline's entries): a price boundary — for a call, one tick below the lowest midpoint of the minute
   before entry; after each completed one-minute midpoint bar (built from the 5-second samples), the low of the latest two
   completed bars minus one tick, raised when higher and never lowered. For a put, mirrored (the high plus one tick,
   lowered only). Early exit when, for 10 continuous seconds of healthy samples, the midpoint is beyond the boundary
   (below it for a call) and the direction state opposes the position (BEARISH for a call); sell at the option's bid then.
   The original time exit stays in force; no extension, re-entry or partial exit.
3. Direction only: at the clock, BULLISH buys a call, BEARISH a put, UNCLEAR no trade; the contract [definition added] is
   the strike nearest 0.10 delta of that right in the clock's recorded chain window and the app's expiry (within 0.12),
   priced from the chain's minute samples (ask at the clock's minute, bid at the clock + 60 min). Primary diagnostic: the
   future's 5-minute forward midpoint move in the chosen direction, with |move| under one tick counted as "small".

Each of the three is computed twice: with the book-pressure condition (the specification) and without it (the control).

## Data
Every clock the app observes from 2026-10-06 to 2026-11-27. Halves: 6–30 October and 2–27 November. 1–5 October is not
used. Columns are written by a script in the hourly ledger; its sha256 is recorded in a dated amendment before any judged
result is computed.

## Verdict
Each comparison PASSES only if, judged once after 2026-11-27: (a) its mean paired difference against the baseline (per
pound of premium, a missed entry counting as 0 for entry and direction) is positive with t > 2, t across days of the daily
mean; (b) at least 200 clocks on at least 15 days (for the exit: at least 40 clocks where it fired); (c) the mean is
positive in both halves; and (d) for entry and direction, the trades it takes average a positive return after the app's
fills, commission and FX (t > 2 across days). Order-book pressure is credited only if, in addition, the specification beats
its no-Level-2 control on the same clocks with t > 2; otherwise any pass belongs to the price-progress rules alone.

Reported regardless: net results after costs, the four-slot portfolio replay, drawdown, coverage (share of clocks entered),
losses avoided and large winners missed or cut short (options that at least doubled), the direction diagnostic's hit rate
and small-move share, by market, by session (09:00–16:00 New York and others) and by expiry (same day, next listed), and
the frozen-scope clocks on their own.

A pass earns a module a place in the frozen method only by a dated rulebook amendment and a new rule version after
2026-11-27.
