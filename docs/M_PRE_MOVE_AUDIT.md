# M and PRE_MOVE_M audit

## Result

The frozen research used a different dollar movement scale `M_price` for each accepted row. The
fixed value `0.475764059845861` was applied only afterwards as a strict threshold on the
dimensionless `PRE_MOVE_M` value. No occurrence was found where that threshold was used as `M`, a
dollar move, or a percentage move.

The research arithmetic is recovered and reproducible. The production source is now frozen by
user decision as IBKR contract-specific Model Option Computation `impliedVol`, tick type 13. The
user accepts the earlier empirical IBKR/EODHD parity conclusion; its numerical artefact is no
longer available and is not a Stage 4 blocker.

## Authoritative lineage

- `2026-09-01-session-hard-structure-d-price-volume/rvol_efficiency_context_v0/`
  `run_experiment.py::derive_pre_move` produces `PRE_MOVE_M`. Its frozen runner SHA-256 is
  `3e0c2884ff3f0277265b86d4feca447954e75a349d3d81388390007e50c47f25`.
- Its accepted `enriched_context_ledger.csv` has SHA-256
  `d253a516be7f3dc1e65a6048679d7d711650706c0df846ef1f6ef29255ecd42e`.
- `20260830-session-hard-broad-universe-expansion-v0/run_broad_experiment.py` produces row-level
  `atm_iv`, `iv_expected_absolute_15m`, and `M_price`.
- Historical commit `2ccb21347b1588a67acedee83b12ce59cb37a008`, file
  `broad_conflict_options_iv_screen_v0.py`, owns the frozen option selection, quality checks, IV
  averaging, and expected-absolute-movement transform.
- `2026-09-01-session-hard-structure-d-price-volume/rvol_efficiency_context_v0/`
  `artifacts/primary/cutoffs.json` records the development median
  `0.47576405984586106`. Later accepted experiments freeze the strict threshold literal
  `0.475764059845861` and apply `.gt(threshold)` only to `PRE_MOVE_M`.

The explicit threshold-misuse search covered the current repository and the complete relevant
worktrees `2026-09-01-session-hard-structure-d-price-volume`,
`2026-09-01-session-hard-pre-move-mid-zone`,
`2026-09-01-session-hard-5-slot-candidate-ranking`, and
`2026-08-15-you-are-working-in-my-existing`, in addition to the 37 reachable and 154 dangling
commits recorded by the preceding lineage recovery. The exact threshold literal produced 55 hits
across 43 files. Every executable hit was a named threshold constant or a strict comparison
against `PRE_MOVE_M`; the other hits were contracts, reports, decisions, or tests describing that
same gate. No hit assigned it to `M_price` or compared it with a raw dollar/percentage move.

## Canonical equations

For the selected exact-prior-session option pair:

```text
ATM_IV = (call_implied_volatility + put_implied_volatility) / 2

expected_absolute_return_15m =
    ATM_IV * sqrt(15 / (252 * 390)) * sqrt(2 / pi)

M_price = P0 * expected_absolute_return_15m

alignment_factor = P0 / raw_1m_open[T0]

raw_PRE_move_price =
    abs(P0 - raw_1m_open[T0 - 3 minutes] * alignment_factor)

PRE_MOVE_M = raw_PRE_move_price / M_price

qualified = PRE_MOVE_M > 0.475764059845861
```

`expected_absolute_return_15m` is the expected absolute value of a zero-mean Gaussian move over 15
trading minutes. It is one 15-minute IV sigma multiplied by `sqrt(2/pi)`; `M_price` is therefore
not one standard deviation. Operations use Python/Pandas binary floating point in the displayed
order with no explicit rounding.

The exact prior-session close is used to select the ATM option pair. It does **not** scale the
dollar `M_price`; current-session `P0` does. Consequently, the research did not produce the final
dollar `M_price` entirely from prior-session inputs.

## P0 and T0

`T0` is the frozen `signal_timestamp`, represented as a timezone-aware UTC timestamp. In the broad
research it is the timestamp of the next native five-minute RTH bar after the completed checkpoint
prefix. The score features use bars only through `checkpoint - 1`.

`P0` is `entry_price`, the native five-minute `open` at that same `T0`. It becomes available at
`T0`, not on the prior session. The one-minute source is split-aligned to it using
`P0 / raw_1m_open[T0]`.

The PRE endpoint at `T0` therefore includes the current opening print. The only earlier endpoint is
the exact one-minute open at `T0-3m`; no post-`T0` value is read. Missing `T0` or `T0-3m` produces
an unavailable value, not interpolation. A live decision cannot know final `P0`, dollar `M_price`,
or `PRE_MOVE_M` before the `T0` opening print arrives.

## Option and IV source contract recovered from research

- Observation session: the exact previous session present in the frozen market-session sequence.
- Underlying reference: the last close of that session's native five-minute RTH data, whose final
  row is the last available bar after filtering timestamps to 09:30–15:55 America/New_York. The
  runner does not independently prove a 15:55 row exists on every session or apply a half-day
  calendar rule.
- Option snapshot: the frozen artefacts contain one timestamp per session at 16:00
  America/New_York.
- Download bounds: strikes from 75% through 125% of previous close; expirations from 7 through 45
  calendar days after the observation date.
- Expiry: nearest eligible expiry, ordered by DTE then date, that has a common call/put strike.
- Pair rank: `abs(log(strike / previous_close))`, descending minimum call/put open interest,
  combined relative spread, call/put IV gap, strike, then contract IDs.
- Pair quality: each IV in `[0.005, 5]`; nonnegative bid; ask not below bid; positive midpoint;
  open interest at least 10 on both legs; per-leg relative spread at most 1; plausible optional
  delta/gamma; expiration not before trade date.
- IV combination: arithmetic mean of the selected call and put `implied_volatility`; there is no
  cross-strike or cross-expiry interpolation.
- Missing/invalid chain or pair: unavailable; no substitute expiry, strike, IV source, or provider.

The EODHD field named `implied_volatility` is authoritative for the research run, but its exact
calculation basis (bid, ask, last, midpoint, or a vendor model) is not recorded. That missing field
lineage is material for IBKR equivalence.

## Representative 20-row audit

Rows are accepted BROAD2025 assessment artefacts. Ten distinct stocks share 2025-02-20; OKLO,
SMCI, HOOD, MRNA, and VST also appear on earlier dates. `Raw PRE` is the split-aligned absolute
dollar numerator above. `Q` applies the unchanged strict threshold.

| Symbol | Session | Prior close | ATM IV | P0 | PRE window UTC | M | Raw PRE | PRE_MOVE_M | Threshold | Q |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| OKLO | 2025-01-03 | 21.844999 | 1.081600 | 24.719999 | 15:07–15:10 | 0.263553170 | 0.203999992 | 0.774037328 | 0.475764060 | Y |
| MRNA | 2025-01-07 | 42.560001 | 0.733450 | 47.430000 | 14:57–15:00 | 0.342907319 | 0.320000000 | 0.933196763 | 0.475764060 | Y |
| OKLO | 2025-01-07 | 29.980100 | 1.343750 | 30.229999 | 14:57–15:00 | 0.400414437 | 0.039999999 | 0.099896495 | 0.475764060 | N |
| VST | 2025-01-08 | 163.380004 | 0.617100 | 154.350006 | 15:17–15:20 | 0.938891299 | 0.010000000 | 0.010650861 | 0.475764060 | N |
| SMCI | 2025-01-10 | 32.630001 | 0.841000 | 33.349998 | 14:57–15:00 | 0.276468035 | 0.281515571 | 1.018257214 | 0.475764060 | Y |
| VST | 2025-01-14 | 162.130004 | 0.594850 | 172.250000 | 14:57–15:00 | 1.009996372 | 2.185000000 | 2.163374108 | 0.475764060 | Y |
| MRNA | 2025-01-22 | 35.889999 | 0.644600 | 40.869998 | 15:07–15:10 | 0.259685700 | 0.034999998 | 0.134778304 | 0.475764060 | N |
| SMCI | 2025-01-22 | 32.439998 | 1.036600 | 34.849998 | 15:07–15:10 | 0.356095987 | 0.149999991 | 0.421234715 | 0.475764060 | N |
| HOOD | 2025-02-12 | 53.349998 | 1.032800 | 55.249801 | 14:57–15:00 | 0.562470907 | 0.159800003 | 0.284103588 | 0.475764060 | N |
| HOOD | 2025-02-13 | 55.959999 | 1.114100 | 62.025001 | 14:57–15:00 | 0.681152055 | 0.200300003 | 0.294060631 | 0.475764060 | N |
| AXON | 2025-02-20 | 593.309997 | 0.770350 | 540.525024 | 15:07–15:10 | 4.104469273 | 0.563700025 | 0.137338103 | 0.475764060 | N |
| HOOD | 2025-02-20 | 59.220001 | 0.587900 | 55.549999 | 14:57–15:00 | 0.321914569 | 0.529999990 | 1.646399513 | 0.475764060 | Y |
| IVZ | 2025-02-20 | 18.200000 | 0.305750 | 17.975000 | 15:07–15:10 | 0.054173698 | 0.075000000 | 1.384435674 | 0.475764060 | Y |
| LW | 2025-02-20 | 57.150001 | 0.381700 | 57.409999 | 16:07–16:10 | 0.216004509 | 0.034999999 | 0.162033652 | 0.475764060 | N |
| MRNA | 2025-02-20 | 35.890098 | 0.717250 | 34.470001 | 16:27–16:30 | 0.243705298 | 0.030000001 | 0.123099502 | 0.475764060 | N |
| NCLH | 2025-02-20 | 27.010000 | 0.763000 | 24.600000 | 14:57–15:00 | 0.185017480 | 0.000000000 | 0.000000000 | 0.475764060 | N |
| OKLO | 2025-02-20 | 45.110000 | 1.147500 | 40.965000 | 14:57–15:00 | 0.463360246 | 0.200000000 | 0.431629605 | 0.475764060 | N |
| SMCI | 2025-02-20 | 60.269901 | 2.044100 | 57.130001 | 15:27–15:30 | 1.151117051 | 0.320000006 | 0.277990848 | 0.475764060 | N |
| UAL | 2025-02-20 | 106.500000 | 0.467100 | 102.214996 | 14:57–15:00 | 0.470627825 | 0.454999982 | 0.966793628 | 0.475764060 | Y |
| VST | 2025-02-20 | 169.339996 | 0.916100 | 159.770004 | 14:57–15:00 | 1.442750543 | 0.835700021 | 0.579240829 | 0.475764060 | Y |

Recalculation against the selected option records and underlying Parquet bars matched stored `M`
within `3.6e-15` and stored `PRE_MOVE_M` within `1.2e-14`. On 2025-02-20, `M` ranges from
`0.054173698` (IVZ) to `4.104469273` (AXON). Each repeated stock has different `M` values across
the audited dates.

## Stage ownership established by causality

Stage 4 owns the qualified IBKR identity, IBKR-only persistent data substrate, exact-prior-session
underlying/option context once its IBKR field contract is proven, and the dimensionless
`expected_absolute_return_15m` derived from prior-session ATM IV.

Stage 5 owns the current-session signal checkpoint, `T0`, `P0`, final dollar `M_price`, raw PRE
movement, `PRE_MOVE_M`, its fixed threshold, cohort percentiles/bands, qualification, and ranking.
This is required by the recovered code: `M_price` cannot be finalised until current-session `P0`
arrives at `T0`.

No Stage 5 runtime implementation is added by this audit.

## IBKR production mapping

- Previous-session underlying: calendar-complete XNYS `5 mins` `TRADES` bars with `useRTH=True`,
  ending at the prior session close. A fully absent session uses `duration=1 D`; partial cache gaps
  request only contiguous missing ranges. This is an explicit IBKR implementation mapping of the
  research's five-minute RTH traded-price context. The final scheduled bar close is the
  option-selection reference; any missing scheduled bar makes the context unavailable.
- Option source: exact qualified SMART call/put contracts. Contract-specific
  `ticker.modelGreeks.impliedVol` is IBKR Model Option Computation tick type 13. Tick types
  10/11/12 and generic tick 106 are excluded as IV sources; generic tick 101 is used only for
  per-leg open interest.
- Timing: uncached model context is captured after the observation-session close and before the
  target-session open, then persisted. A late request cannot reconstruct historical option model
  ticks and fails closed.
- Failure: any missing underlying bar, option leg, qualifying identity, quality input, or tick-13
  model IV returns `PRE_CONTEXT_NOT_READY`. There is no fallback provider or fallback IV field.

The earlier user-run vendor comparison found IBKR model IV practically equivalent to the EODHD IV
used in the frozen research. That conclusion is accepted; the old CSV/report is unavailable, so no
bit-for-bit vendor equality or new parity research project is required. Runtime data is IBKR-only.

## Test boundary

`tests/test_pre_move_m_research_contract.py` freezes six representative input/output rows and
checks row-specific `M`, repeated-stock time variation, cross-sectional variation, the exact
normalisation arithmetic, current `P0` rather than prior-close scaling, and threshold placement.
These remain reference arithmetic tests. `tests/test_stage4_pre_context.py` additionally freezes
the pure Stage 4 ATM-IV/expected-return calculation, exact option selection, tick-13-only source,
fail-closed model-IV behavior, `conId` lineage, persistence reuse, and absence of Stage 5 fields.
