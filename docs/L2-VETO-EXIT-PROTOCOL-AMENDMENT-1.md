# Amendment 1 to the entry-veto and exit protocol (2026-10-01, before any analysis)

Frozen 2026-10-01 at the user's request, A1-A4 as written; its sha256 is in `L2-VETO-EXIT-PROTOCOL.sha256`. It
governs together with `L2-VETO-EXIT-PROTOCOL.md`, which is unchanged. Nothing below changes the runtime; no
recorded opportunity has been analysed.

## Why
From 2026-10-02 the app records the look14 forecast itself at every clock (`signals.detail.forecast`,
`LOOK14_FORECAST_V1`, commit 792e3cf). V7 and X1 described working it out afterwards, with two details that
differ from both the app and the research the weights came from:
- V7 named a clock profile from all 80 cached days. The weights (0.19 / 0.36 / 0.13 / 0.97) were fitted with the
  profile of days 1-40, the 2026-10-01 live check used days 1-40, and the app records with days 1-40.
- V7 took today's minutes from the app's Saxo bar cache. The app uses the 1,200 Saxo bars it holds at the clock.

One definition for the live screen, the record and the verdict removes the choice.

## Changes
- **A1 Forecast source (V7, X1).** The forecast is the one recorded at the clock in `signals.detail.forecast`:
  status `OBSERVED`, version `LOOK14_FORECAST_V1`, profile `stocker_execution/look14_profile.json` (days 1-40 of
  the cached IBKR bars; sha256 `98dc834c7360…`, stored in full with each record). Nothing is recomputed afterwards.
  A clock without it, or for V7 without a priced option check (`forecast.option` with no `reason`), is excluded
  from that rule and counted, under the existing data rule. This excludes 2026-10-01, recorded before the field
  existed.
- **A2 V7 at the ask, on recorded fields.** The ratio is the movement to expiry implied by the recorded
  `forecast.option.ask` (undiscounted Black on the recorded `futures_mid`, strike and right; the app's
  `forecast.implied_move`) divided by the recorded `forecast_move_to_expiry`. Both span the same time, so this is
  V7's "implied volatility at the ask at most k times the forecast to that expiry". The recorded
  `implied_over_forecast` (at the mid) is reported alongside, not used.
- **A3 V7 grid.** k in {0.8, 0.9, 1.0, 1.1, 1.25}, was {0.8, 0.9, 1.0}. The first live readings on 2026-10-01
  (NQ 0.93, SI 0.79, NG 1.30 at the mid in the afternoon; CL about 1.24 in the morning check) suggest that a grid capped at 1.0 could keep
  fewer than the 150 holdout opportunities a pass needs. This uses price levels seen on one discovery day, never
  any profit or loss. k is still chosen on discovery and judged once on holdout.
- **A4 X1's expected move.** One expected 15-minute move is the recorded `forecast.level` at the clock times the
  square root of the profile's variance for the 15 minutes ending at the check (20 or 30 minutes after the clock).
  The future's move in the trade's favour is the log change from the last completed bar close before the clock
  (`inputs.futures_price`) to the futures mid in the capture at the check (the last quote at or before it).

## Unchanged
Everything else, including the data and P&L rules, the periods (V7 and X1 in effect see discovery from
2026-10-02), the other candidates, the verdict, and the count of thirteen families. No family is added.
