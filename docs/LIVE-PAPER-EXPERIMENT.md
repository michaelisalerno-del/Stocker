# LIVE-data paper experiment: can the order book veto bad clocks?

Question: at the frozen hourly clocks, do the futures order book (Level 2) and simple volatility
ingredients separate the opportunities that later did badly from those that did well, by more than
the trading cost? Research on bars could not answer it; it needs real option quotes and real depth.

## What the runtime records (observation only; no entry rule changes)

Every observed clock, including vetoed, unaffordable and capacity-skipped ones, stores in
`signals.detail`:

- `option_context` and `option_chain`: the selected option and the chain window as seen at the clock
  (existing).
- `book_flow`: the futures book at the clock — spread, depth at 1/3/5/10 levels, depth and
  order-count imbalance, size-weighted midpoint displacement, depth changes over 5/30/60 seconds and
  imbalance persistence (`SAXO_SAMPLED_BOOK_FLOW_V1`; V2 from 2026-10-01 adds traded volume over 5/30/60 s). `UNAVAILABLE` when no depth has arrived;
  never zero-filled.
- `observation`: `rv60`, `session_travel_since_0800` with `session_minutes_counted`, and
  `hour_reference_rv15_median`. `rv15` is in `inputs` when the eligibility gate passed. Gaps leave a
  value unset rather than bridging it.
- `forecast` (from 2026-10-02, `LOOK14_FORECAST_V1`): the look14 forecast from the 2026-09-30 bar
  research, 0.97 × (today so far)^0.19 × (last hour)^0.36 × (last 15 minutes)^0.13, each movement
  against its normal for the time of day. The normal is a frozen per-minute profile
  (`stocker_execution/look14_profile.json`: days 1-40 of the cached IBKR bars, the profile the
  weights were fitted with; its sha256 is stored with each record). `next_hour_move` is the forecast
  standard deviation of the log price over the hold. For the selected option, `option` sets the
  movement to expiry implied by the mid (undiscounted Black) against the forecast's
  (`implied_over_forecast`; above 1, the option is priced for more movement than forecast) and the
  ask against the forecast's fair price. Spans to a later expiry add the profile's full session for
  each weekday between (holidays unknown). Saxo's 1,200 chart bars can begin after the 18:00 open;
  `today_minutes_counted` says how much of the session "today so far" covered. The trade ticket shows
  the same values live for the current candidate.

With persistent capture enabled, the existing recorder also keeps every delivered futures and
candidate-option message from 15 minutes before to at least 60 minutes after each clock. The option
bid/ask at the 60-minute exit therefore exists for every opportunity, traded or not; the four-position
limit does not reduce the sample. Paper trades buy one contract whatever it costs, up to a £1,000 guard
against a bad quote (was £50).

From 2026-10-02 (the user's request, to study other strikes) the observation-only option chain window
holds the 25 strikes nearest the money (`option_chain_strikes: 25`) on the held or candidate option's
expiry, moving once the money leaves its middle half, and every chain message (calls and puts: bid, ask,
sizes, Saxo's Greeks and bid/ask/mid volatility, volume, open interest) is recorded as its own instrument
(`asset_type` `OptionsChain`, keyed by the future's UIC) in each clock's capture, with a checkpoint of the
merged chain (rows keyed by `Index`) before its retained messages. The candidate's regular quote stream is
unchanged; its per-option `CHAIN_CONTEXT` rows now appear only while its strike is inside that window. The
capture allowance is 20 GiB (`archive_max_bytes`; was 2 GiB, about a week) and the shared 15-minute memory
64 MiB.

## Operator steps (user actions are marked)

1. **User:** create a LIVE Saxo OpenAPI application with the redirect URI
   `https://139.59.178.164/oauth/saxo/callback`; keep the client id and secret off chat and Git.
2. **User:** subscribe the Saxo account to real-time CME Group data (CME, NYMEX, COMEX) including
   Level 2 / market depth, and enable it for API use. Saxo sends real-time prices only to the
   user's one FullTradingAndChat ("primary") session, and entries need it. SLRNO never takes it
   itself: the user clicks **Use real-time in SLRNO** on System (`POST /api/session/primary`),
   which renews the price streams. It can delay or log off SaxoTraderGO, and a SaxoTraderGO login
   takes it back, so click again after each one.
3. Server: write `/etc/stocker/v1/saxo.live.credentials.json` (0600, owned by the service user,
   `"environment": "SAXO_LIVE"`), copy `configs/saxo.live.paper.example.yaml` to
   `/etc/stocker/v1/saxo.live.paper.yaml`, and point the unit at a **new** ledger
   `/var/lib/stocker/v1/saxo-live-internal-paper.sqlite3`. Start disarmed.
4. **User:** System → Connect Saxo; then select the LIVE AccountKey in the credentials file; restart.
5. Pin one verified contract per market from the LIVE discovery candidates (`contracts`), and
   supply the prior-session reference selection audit (`reference_selections_file`).
6. **User approval:** per-market option mappings (`mappings`): option root, delta tolerance, fee
   evidence and expiry-time evidence. Unapproved markets stay monitor-only but are still recorded.
7. **User:** put the actual recording-permission record in `recording_permission_evidence` and set
   `persistent_capture: true`.
8. **User:** click **Use real-time in SLRNO**. Verify on System that the session is
   FullTradingAndChat, L1 is real-time (`DelayedByMinutes` 0), depth fields arrive and option
   quotes update; run `POST /api/paper/preflight`; arm with `ENABLE PAPER ONLY`. Every restart
   disarms.

## Analysis (after the recording, not before)

Collect 6-8 weeks (about 300-500 clocks per market that has an approved mapping). Per opportunity:
option bid/ask at the clock and at +60 minutes from the capture archive, the recorded `book_flow`
and `observation`. Choose any veto on the first half of the weeks only, judge it once on the second
half, and count a veto as useful only if the kept trades beat all trades by more than their round-trip
cost with t > 2. The bar-based research expectation is that they will not; this records the evidence.

The exact candidates, periods and pass bar were frozen on 2026-10-01 in
[L2-VETO-EXIT-PROTOCOL.md](L2-VETO-EXIT-PROTOCOL.md) (seven entry vetoes, six exits; discovery 1-30 October,
holdout 2-27 November); that file governs the analysis, with
[Amendment 1](L2-VETO-EXIT-PROTOCOL-AMENDMENT-1.md) (also 2026-10-01: V7 and X1 use the recorded `forecast`) and
[Amendment 2](L2-VETO-EXIT-PROTOCOL-AMENDMENT-2.md) (2026-10-02: W1 waits out a blown-out option quote at entry and
exit, up to two minutes; judged on 2 October-27 November against a placebo). A separate frozen
test, [RELEASE-EXIT-PROTOCOL.md](RELEASE-EXIT-PROTOCOL.md), judges selling a minute before a scheduled release that
falls on the exit; its calendar `release-calendar-2026-10-11.yaml` is also the app's `event_calendar_file` from
2026-10-02, so each recorded clock lists the releases near it. [MINUTE-EARLY-EXIT-PROTOCOL.md](MINUTE-EARLY-EXIT-PROTOCOL.md)
judges selling one minute before the hour on all other exits. [RELEASE-ENTRY-PROTOCOL.md](RELEASE-ENTRY-PROTOCOL.md) judges
deciding and buying one minute before a release that falls on the clock. [RELEASE-LATE-ENTRY-PROTOCOL.md](RELEASE-LATE-ENTRY-PROTOCOL.md)
judges keeping the decision and buying one minute after that release. [QUOTE-LEAN-PROTOCOL.md](QUOTE-LEAN-PROTOCOL.md) judges
whether the side a blown-out option quote leans to (bid falling or ask jumping) points the future's next 5 minutes. [EXIT-SET-2-PROTOCOL.md](EXIT-SET-2-PROTOCOL.md)
judges a futures trailing exit (E1) and a buyers-and-sellers-turn exit (E2) on 5-16 October, and selling into the
09:30/10:00 New York burst (E3) by 27 November. [WALL-PROTOCOL.md](WALL-PROTOCOL.md) judges whether big
resting orders in the futures book make their price less likely to be reached within 5 minutes, on 5-16 October.
[INTRADAY-MOMENTUM-PROTOCOL.md](INTRADAY-MOMENTUM-PROTOCOL.md) judges, on paper from the app's NQ minute bars, betting the
last half hour in the direction of the first (Gao, Han, Li & Zhou), on 5 October 2026 to 15 January 2027.
