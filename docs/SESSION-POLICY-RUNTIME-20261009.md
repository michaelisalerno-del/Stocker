# Session policy in the runtime, alongside the hourly trades (2026-10-09; the user's decision: "use the best method we've discovered so far in my app and continue recording")

## What changes
- The first clock of each CME session (19:00 New York; there is no 18:00 clock) becomes the SESSION trade: the app buys its
  normally selected option (the overnight rule selects the next day's expiry) and holds it, instead of 60 minutes, until
  09:25 New York on the next weekday — or earlier, on an armed trail that arms when a current bid reaches 1.50 x the entry
  price and sells at the first later current bid at or below 0.60 x the highest bid since entry. An empty bid never
  triggers a sale (the scheduled exit still applies). The option subscription is kept until the exit.
- Direction (amended 2026-10-09, the user's "Flip it"): the SESSION trade buys CALLS in all four markets. The overnight
  drift is up in ES, NQ and GC (ETF 2012–2026: +2.8 / +4.6 / +2.8 bp a night, t 2.5 / 3.5 / 2.2; futures 2024–26: up on
  57–58% of nights, so the puts were right on 42–43%); an option model of 547 nights puts calls 10–15 points per trade above
  puts in ES/NQ/GC and level in CL (which already buys calls). The day does not flip the overnight drift. The 19:00 clock's
  selected option is therefore a call; frozen protocols that read the app's option at every clock report the 19:00 clock
  separately from this date. Hourly and DAY trades keep the frozen directions (CL calls; ES/GC/NQ puts).
- The 10:00 New York clock becomes the DAY trade (the user's "one trade out of hours and one in hours", 2026-10-09): the
  end of the open's first pullback (the futures anatomy in STUDY_LOOK.md: the first move is retraced 70–109% by ~10:00–10:30
  and a second push of about the same size follows). It buys the NEXT trading day's expiry (never the same day's, whose
  entries lose a median 93–95% at every time of day), holds to 11:00 New York (changed from 15:00 the same day, on the user's "Change it": in a model of ~383 days per
  market 11:00 beat 15:00 on 68–77% of days, with less time decay and a cheaper exit toll), or sells earlier on the same 1.50 / 0.60
  trail. The event carries `expiry_after` (the clock day's 17:00 New York close) and the selector takes the first expiry
  after it. Untested on recordings (the following-expiry capture is sparse in US hours); this is what produces the data.
  One veto, ES and NQ only (`DAY_OPEN_AGAINST`): no day trade when every one-minute close from 09:30 to 09:59 New York
  sat at least 10 bp against the market's direction relative to the 09:30 open (on ~400 ETF days those legs lost 23–26 bp
  from 10:00 to 15:00; GC and CL showed nothing; it fires on roughly one day in eight). A missing bar means no veto. The
  clock still selects and records its option; the veto is recorded as the skip reason.
  Cost gate (`DAY_SPREAD_ABOVE_GATE`, all markets): the day trade buys only when (ask − bid) / ask ≤ 3% at entry. In the
  option model the 10:00→11:00 leg breaks even at a round-trip toll of about 3.1% (ES), 3.4% (NQ), 3.8% (GC), 8.8% (CL);
  observed tolls are ~2–4% ES, ~4–7% NQ, ~9–16% GC, ~14–25% CL, so in practice it trades mostly ES, sometimes NQ.
- Every other clock trades exactly as before (60-minute hold; the armed trail of EXIT-SET-3 rule AT, 1.20 / 0.75), so the
  hourly incumbent keeps producing its fills and the frozen hourly tests keep their data. Selection, recording and the
  chain views are unchanged at every clock. The event carries `policy` ("SESSION" or "HOURLY") and `trail` (arm, keep).
- Paper only (INTERNAL_PAPER); no configuration change; RULE_VERSION unchanged (the clock set and the frozen definitions
  are unchanged; the 19:00 clock's exit and trail are the only differences, and they are recorded on the event).

## Why (the day's evidence, `~/Codex/2026-10-09-claude-codex-entries-exits/DEFINITIVE_PLAN.md`, `STUDY_LOOK.md`)
The loss is the toll (entry −7.9, mid-to-mid −0.6, exit −7.4 points per £ of premium, 269 trades); the direction never
changes per market; one option per session beat the sum of the hourly trades in 84–90% of sessions; an open-bought option
is worth most around 09:25 and bleeds through the day as its exit toll climbs; bigger moves give back less, so the
60-minute trail (0.75) is too tight for a session. None of this is a proven edge; it is the lowest-toll version of the
same bet, run live in paper so the study (`STUDY_DESIGN.md`) has its own prospective data.

## Effect on the frozen tests (stated as an amendment, dated 2026-10-09, before any judged result)
- Protocols judged on recorded quotes at every clock: unaffected (every clock still selects and records its option).
- Variants that use the app's own fill at a clock: the 19:00 clock's hourly fill no longer exists from the first session
  after deployment; those variants are judged on the remaining 20 clocks a day, and the 19:00 clock is reported separately.
- The session trade's own results are judged by STUDY_DESIGN.md (G1 x X4 against the hourly incumbent, first-trade-then-
  cash and the full-session trail), not by any hourly protocol.
