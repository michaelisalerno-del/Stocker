# Frozen futures execution rulebook — CLOCK60_NG13_20260927

## Authority and reconciliation

The frozen primary source is `research/futures-integration-sources/fixed_spec.json`, an exact
copy of the accepted FIXED_ARCHITECTURE_SPEC from September 26. The manifest records original
paths and SHA-256 hashes, including build_features.py and tail_first_v1/tail.py. Prior user
instructions in task 01a0df73 explicitly retained only NG CLOCK_13 for the current replay.
The present integration prompt authorises PAPER implementation with the small-size limits.
It does not approve a new listed-product/strike/expiry adaptation.

CURRENT_RESEARCH_NG13_ONLY is the fixed-clock primary, not the stopped directional-trigger
family. R4, G7, fresh-delta, failure-gate exits and all new volume vetoes are excluded.
FAST_GC16_G2_DECONFOUND_V0 is diagnostic only. FAST_BTC_REAL_EXPIRY_RECONSTRUCTION_V0 rejects
17:00 NY as a real BTC expiry and concludes real listed mapping remains inconclusive.

## Per-market frozen specifications

| Market | Frozen source primary | Direction | Target absolute delta | Opportunities | Veto | Order readiness |
|---|---|---|---|---|---|---|
| BTC | tail_first_v1/BTC/TAIL_FROZEN.json; fixed_spec | call_10d_0DTE / bullish | 0.10 | hourly 09–16 NY weekdays | none | Product, real expiry adaptation and tolerance unapproved; BTC/MBT/BFF never interchangeable |
| CL | tail_first_v1/CL/TAIL_FROZEN.json; fixed_spec | call_10d_0DTE / bullish | 0.10 | hourly 09–16 NY weekdays | none | Listed product, real expiry adaptation and delta tolerance unapproved |
| GC | tail_first_v1/GC/TAIL_FROZEN.json; fixed_spec | put_10d_0DTE / bearish | 0.10 | hourly 09–16 NY weekdays | none | Listed product, real expiry adaptation and delta tolerance unapproved |
| NG | tail_first_v1/NG/TAIL_FROZEN.json; fixed_spec | put_10d_0DTE / bearish | 0.10 | hourly 09–16 NY weekdays | exact 13:00 NY purchase excluded | Listed product, real expiry adaptation and delta tolerance unapproved |
| NQ | tail_first_v1/NQ/TAIL_FROZEN.json; fixed_spec | put_10d_0DTE / bearish | 0.10 | hourly 09–16 NY weekdays | none | Listed product, real expiry adaptation and delta tolerance unapproved |
| SI | tail_first_v1/SI/TAIL_FROZEN.json; fixed_spec | put_20d_0DTE / bearish | 0.20 | hourly 09–16 NY weekdays | none | Listed product, real expiry adaptation and delta tolerance unapproved |

Every row uses rule version CLOCK60_NG13_20260927 and the common mechanics below. No market
is execution-ready merely because its underlying has bars. Missing authority is reported as
LISTED_PRODUCT_AND_DELTA_TOLERANCE_UNAPPROVED. No micro product substitution is approved.

## Common frozen mechanics

Signal FUT and purchased FOP identities remain separate. Signal monitoring chooses the highest
volume nearby actual future on the strictly preceding completed exchange session, excluding
contracts at/after last-trade date; equal-volume ties use conId ascending. Candidate metadata is
bounded to the six nearest listed contracts. Missing prior-session volume blocks selection.
The six-contract bound and conId tie-break are documented execution conventions; the frozen
research said “nearby” without an ongoing numerical bound. Each source reference date uses
its own selected contract, including subsequently expired contracts. No continuous symbol or spliced
return is used. A rollover starts a new history buffer; positions retain their original FOP.

Bars are UTC minute-start-stamped, available only after the minute completes. The purchase
input is the close at opportunity minus one minute. RV15 is the square root of the sum of 15
squared completed log returns (16 closes). Eligibility requires 31 contiguous completed closes,
positive RV15, five preceding source-session hourly median references, the inherited finite
feature gate and next weekday strictly before the future last-trade date. Session feature
reference starts 08:00 NY, including the original opening-range and cumulative volume window.
There is no interpolation, forward-fill, zero-RV floor or historical signal replay after restart.
The runtime fetches only the bounded history required for these references, then streams bars.

The opportunity and purchase clock are identical: exact hours 09:00 through 16:00 America/New_York
on weekdays. There is no newly invented price crossing or smooth trigger gauge. The only veto
is NG 13:00. Purchase direction is fixed by the table. Exit is **original opportunity +60 minutes**,
not actual fill +60. Later order/fill/diagnostic timestamps are separate evidence.

The historical strike formula used Black76, r=0, sigma=RV15*sqrt(525600/15), ACT/365 and
K=F*exp(0.5*sigma²*T - sign*NormalInverse(abs_delta)*sigma*sqrt(T)), with a hypothetical
same-day 17:00 NY expiry. That synthetic expiry and continuous strike cannot be sent to IBKR.
No historical synthetic option price is an executable quote.

## Explicit execution adaptations and safeguards

A market's ProductMapping must cite an approved source and execution adaptation, actual product,
symbol, exchange, trading class, currency, multiplier, price magnifier/units, termination zone/time,
settlement mechanism, conservative fee reserve and delta tolerance. No mappings are supplied by
this change because those missing choices cannot be inferred from research labels.
The implemented adaptation selects the nearest listed **frozen-model delta**, using the approved
real expiry. This changes synthetic strike construction and requires explicit mapping approval;
it does not silently use broker IV/fresh-delta or seek a cheaper strike. Adjacent listed strikes
are compared deterministically; lower strike wins an exact tie. Out-of-tolerance means skip.

Broker details must confirm actual FOP conId, underlying conId, exchange, tradingClass, multiplier,
currency, expiry and termination time. Listed expiry must be today and still live. No match is
NO_REAL_0DTE_MATCH. No different expiry or direct future may replace it. Exit plus a 120-second
operational buffer must precede both termination and the end of the actual option trading session;
otherwise UNSUPPORTED_EXIT_BEFORE_CONTRACT_CUTOFF. This is an execution support check, not a new exit.

Fresh positive uncrossed real-time bid/ask from the current request is required (five-second age).
Delayed/frozen or missing prices block entry. One entry LMT at the ask rounded down to a verified
market-rule increment; it expires at opportunity+20 seconds and is actively cancelled at deadline.
There is no entry price chase or resubmission after timeout. A late fill keeps its original exit.
At minute60 submit a SELL LMT at a fresh bid; maximum three attempts, each 20 seconds, only after
reconciliation of the previous attempt. A failed exit is surfaced as exposure requiring operator
attention. Exercise/delivery or unexplained positions block new entries and are never erased.

Admission order: opportunity UTC ascending, then BTC, CL, GC, NG, NQ, SI. No stale queue. Each
opportunity has one persistent identity per market, signal conId, clock and rule version. There is
no same-event re-entry. Later valid clocks can enter after prior closure; overlapping trades retain
separate obligations and cannot exceed global limits. There is no lifetime or daily consumed slot.

Quantity is exactly one. Complete premium = limit * verified multiplier * price-unit factor.
Convert USD to GBP with a timestamped GBPUSD bid (maximum 30 seconds), add conservative fees and
round cash upwards to pennies. Above £10 is SKIP_BUDGET_TOO_SMALL. No fractional quantity, larger
budget, alternative strike/expiry/product or multiple cheap contracts. Atomically reserve a full
£10 and one of four slots before durable submission intent. Pending, uncertain, partial and closing
exposure retain reservations until matching executions, terminal orders and broker positions agree.
£40 is a concurrent allocation ceiling; cumulative daily losses may exceed it.

## GC and BTC

GC retains its base fixed-clock put and minute60 management. G2, GC16 exclusion/early exit and
volume vetoes have no execution switch or order dependency. Completed OHLCV, range3 and volume
inputs are retained as observations. A 16:00 cohort label is never a later failure-trigger timestamp.
The card says “Experimental management disabled”.

BTC monitoring is independent of equity sessions/weekends. Actual broker contract calendars govern
market status. CME expanded cryptocurrency futures/options to 24/7 from May 29, 2026, with maintenance
and following-business-day trade dates for weekend activity ([CME notice](https://www.cmegroup.com/notices/electronic-trading/2026/05/20260525.html)).
Monitoring does not extend the frozen weekday entry clocks. Standard BTC/MBT expiry is 16:00 London;
BFF expiry is 16:00 New York in the audited sources. UK/US DST differences are calculated by zone,
not fixed offsets. The reconstruction's standard/BFF switch was a diagnostic, not a product approval.
At/after a product's same-day termination there is NO_REAL_0DTE_MATCH; no synthetic 17:00 substitution.
A minute60 endpoint at termination is unsupported because it cannot safely close before the cutoff.

## Accounting and evidence

Broker executions and reported commissions drive realised results. Missing costs or execution FX
leave results provisional; missing quotes never close a trade. Unrealised bid valuations show quote
time and freshness and remain estimates. Historical research, prospective observations and broker
PAPER executions remain separate; the fresh ledger begins with no historical profits.

Every decision records source/rule identity, original clock, inputs, actual future/FOP, quote times,
model delta, budget, FX, fees, capacity, order references, fills, commissions and skips. SQLite WAL,
FULL synchronous writes, durable intent, process ownership and startup reconciliation remain required.
