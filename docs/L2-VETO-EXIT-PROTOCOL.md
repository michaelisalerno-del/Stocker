# Entry vetoes and exits from the live recordings — frozen protocol (2026-10-01, before any analysis)

The user chose to fix these candidates now and judge them later on the LIVE paper recordings; the app keeps trading
the frozen method unchanged (hold 60 minutes) as the baseline. Nothing below changes the runtime. The hash of this
file is in `L2-VETO-EXIT-PROTOCOL.sha256`; any later change is an amendment, dated and stated as such, made before
the holdout is looked at.

## Data
- Opportunities: every recorded clock from 2026-10-01 with a verified candidate option, traded or not (vetoed,
  unaffordable, capacity-skipped and quote-stale clocks included when the option has quotes at the clock and at the
  exit). Sources: `signals.detail` (`inputs`, `observation`, `book_flow`, `option_context`, `option_chain`) and the
  capture segments (every delivered futures and candidate-option message from 15 minutes before to at least 60
  minutes after the clock, each futures update with its `book_flow`).
- Periods: DISCOVERY = clocks 2026-10-01 to 2026-10-30; HOLDOUT = clocks 2026-11-02 to 2026-11-27. Analysis runs
  once, after 2026-11-27. Contract re-pins (CL about 12-15 Oct, NG about 20-23 Oct) do not split the data.
- P&L per opportunity, exactly the app's internal fills: buy one contract at the recorded ask plus one tick at the
  clock (the first quote within the 20-second entry deadline), sell at the bid less one tick at the exit time (the
  last quote at or before it); $2 commission plus the exchange fee per side from the recorded trading conditions;
  GBP at the recorded GBPUSD with Saxo's conversion mark-up each way (0.6% on this account). Measured as a share of
  the money paid in. For traded clocks the actual fills are used.
- An opportunity lacking the field a rule needs (book_flow not CURRENT, missing lookback) is excluded from that
  rule's comparison only, and counted.
- Baseline: all opportunities, exit at 60 minutes.

## Entry vetoes (a veto removes opportunities; each threshold is chosen on discovery from the listed grid)
"Normal" levels are per market and New York clock hour, taken from discovery only.
- V1 Thin book: skip when the futures spread at the clock is above its normal median, or 10-level depth (bid + ask)
  is below its normal 25th percentile. Grid: spread rule only / depth rule only / either.
- V2 Imbalance against the trade: skip a call when the 5-level imbalance is below -x, a put when it is above +x.
  x in {0.10, 0.20, 0.30}.
- V3 Book draining: skip when 10-level depth changed by less than -y of its level over the 60 seconds before the clock
  (`lookbacks.60.10` bid_change + ask_change, divided by the depth then). y in {10%, 25%}.
- V4 Expensive option: skip when the option's spread at the clock is above z of its mid. z in {10%, 20%, 30%}.
- V5 Calm book: skip when 10-level depth is at or above its normal 75th percentile and |5-level imbalance| < 0.10.
- V6 Fragile book only: keep only clocks whose spread is at least twice its normal median or whose 10-level depth is
  at or below its normal 25th percentile.
- V7 Cheap against the forecast: keep only clocks where the option's implied volatility at the ask (Black-76, time to
  its verified expiry instant) is at most k times the look14 forecast to that expiry (clock normal x today so far^0.19
  x last hour^0.36 x last 15 minutes^0.13 x 0.97; clock profile from the 80 cached days of
  `2026-09-30-futures-bars-open-look/bars.parquet`; today's minutes from the app's Saxo bar cache). k in {0.8, 0.9, 1.0}.

## Exits (on the recorded option bid path; each replaces the 60-minute exit when it triggers first)
- X1 Time stop: at 20 or 30 minutes, exit unless the future has moved in the trade's favour by at least one expected
  15-minute move (the forecast level x the clock profile). Grid: 20 / 30 minutes.
- X2 Price stop: exit when the bid is at or below half the entry ask.
- X3 Take profit: exit when the bid is at least (1 + p) x the entry ask. p in {0.5, 1.0, 2.0}.
- X4 Trailing: once the bid has reached 2x the entry ask, exit when it has given back half of the gain above the entry.
- X5 Depth-drain exit: once the bid is above the entry ask, exit when 10-level futures depth falls at least 25% below
  its trailing 5-minute average.
- X6 Imbalance-flip exit: once the bid is above the entry ask, exit when the 5-level imbalance averaged over the last
  60 seconds is against the position (call: below -0.10; put: above +0.10).

## Verdict
- Each family's parameter is chosen on DISCOVERY by the largest mean improvement over the baseline after costs; it
  is then judged once on HOLDOUT.
- A candidate PASSES only if, on HOLDOUT, (a) its mean improvement over the baseline (per opportunity, after costs)
  is positive with t > 2, t across days of the daily mean improvement; (b) at least 150 holdout opportunities remain
  (vetoes) or are affected (exits); and (c) its sign of improvement was the same on discovery.
- Thirteen families are tested, so about one false pass is expected by chance at t > 2; a pass at t < 3 is reported
  as weak. Separately reported for every candidate: whether the strategy with it is profitable after costs on
  holdout, by market, by clock hour, winners, worst and best trade, and the share of opportunities affected.
- Combinations are exploratory only, unless a new frozen protocol is written before new data.
