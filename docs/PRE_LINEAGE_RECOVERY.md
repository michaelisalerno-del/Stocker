# PRE lineage recovery

## Result

**BLOCKED — PRE CONTRACT NOT RECOVERABLE**

The accepted research lineage contains an authoritative calculation named `PRE_MOVE_M`. It does
not contain an authoritative bars-only **PRE-level** calculation or the IBKR request contract that
Stage 4 was asked to freeze. `PRE_MOVE_M` also depends on an upstream prior-session ATM-option-IV
movement amount, not only on historical underlying bars. Treating it as the missing Stage 4 PRE
level, or translating its EODHD inputs to IBKR request settings, would require new trading rules.

Stage 4 therefore remains incomplete. The existing contract-independent IBKR cache substrate is
unchanged, no PRE calculator/version has been declared, and Stage 5 has not been started.

## Search scope

The recovery inspected:

- the current source, tests, configuration, documentation, notebooks, fixtures, reports, and
  tracked research artefacts;
- all current-repository branches and remote-only refs, all 37 reachable commits, reflogs,
  deleted/renamed paths, 154 dangling research commits, and the local-path origin (there are no
  tags);
- sibling research worktrees containing the accepted Session HARD, Structure D, PRE_MOVE,
  candidate-ranking, prospective-recorder, and broad-universe runs;
- Parquet/CSV/JSON schemas, manifests, receipts, hashes, and representative stored rows;
- the upstream option-pair selection and IV movement code at historical commit
  `2ccb21347b1588a67acedee83b12ce59cb37a008` (`research: add prior-close options iv movement
  screen`).

Exact and variant searches included `PRE`, `PRE_MOVE`, `pre_level`, `M_price`, Session HARD,
Structure D, first touch, rank ledger, cohort, percentile, and LOW/MID/HIGH. The similarly named
SLRNO scenario/pre-move diagnostic code was rejected because it is a separate, diagnostic-only
breakout-context calculation with no lineage to the accepted Stocker research.

No reachable or dangling version of this repository contains an executable/frozen Stocker
PRE-level implementation, `PreHistorySpec`, or PRE-level input/output fixture. The current
architecture documents are the first occurrences of that proposed bars-only runtime boundary.

## Backward lineage from accepted research

The relevant accepted chain is:

1. `2026-09-01-session-hard-pre-move-mid-zone/contract.json` freezes the qualifying rule
   `PRE_MOVE_M > 0.475764059845861` and consumes the context ledger by SHA-256. This is a
   downstream threshold/band experiment, not the producer.
2. `2026-09-01-session-hard-structure-d-price-volume/rvol_efficiency_context_v0/contract.json`
   defines `PRE_MOVE_M`; its `run_experiment.py` contains the executable producer. The frozen
   runner SHA-256 is
   `3e0c2884ff3f0277265b86d4feca447954e75a349d3d81388390007e50c47f25` and the accepted
   `enriched_context_ledger.csv` SHA-256 is
   `d253a516be7f3dc1e65a6048679d7d711650706c0df846ef1f6ef29255ecd42e`.
3. `pre_admission_price_volume_v0/run_experiment.py` establishes the adjacent causal
   pre-admission window: exact one-minute bars at `T0-3m`, `T0-2m`, and `T0-1m`. That runner
   calculates volume/efficiency context, not `PRE_MOVE_M`; it must not be confused with the
   PRE_MOVE producer.
4. The accepted ledgers supply frozen `P0` and `M_price`. The broad-universe producer at
   `research/directional-readiness/20260830-session-hard-broad-universe-expansion-v0/`
   calculates `M_price` from exact-prior-session ATM option IV.
5. Historical commit `2ccb21347b1588a67acedee83b12ce59cb37a008`, file
   `packages/stocker_research/src/stocker_research/broad_conflict_options_iv_screen_v0.py`,
   contains the option-pair selection and IV movement implementation. The later frozen Structure
   D lineage names commit `98ddf3e6340ebfe854e295ef1a8dcc092127a14e` as its lineage root.
6. The prospective schema explicitly says this prior-session value comes from an **external
   Group-O package producer** and marks missing/invalid values `UNKNOWN_INCOMPLETE`. Its own audit
   says IBKR-versus-EODHD bar construction still requires parallel validation; no verified
   equivalent cases existed.

The cohort percentile, LOW/MID/HIGH band, Session HARD threshold, first-touch rule, and ranking
are downstream consumers. They are not PRE-level mathematics and belong outside Stage 4.

## Confirmed recovered behavior

The following statements are backed by the frozen executable research and artefacts. They describe
`PRE_MOVE_M`; they do **not** establish the requested production PRE-level contract.

| Field | Confirmed behavior | Evidence |
|---|---|---|
| Output | `PRE_MOVE_M` is a dimensionless movement ratio. No collection of PRE price levels is emitted. | `rvol_efficiency_context_v0/contract.json`, `derive_pre_move` |
| Underlying bars | One-minute underlying `open` values are used. Timestamps are parsed as UTC. Duplicate timestamps retain the last row. | `derive_pre_move` |
| Causal points | Exactly `T0-3 minutes` and `T0` are read. No later bar is used. The current `T0` open is deliberately included. | `derive_pre_move` |
| Split alignment | `factor = P0 / raw_open[T0]`; the `T0-3m` open is multiplied by that factor. | `derive_pre_move` |
| Formula | `abs(P0 - raw_open[T0-3m] * (P0 / raw_open[T0])) / M_price` | frozen contract and runner |
| `M_price` | `P0 * atm_iv * sqrt(15 / (252 * 390)) * sqrt(2 / pi)` | broad-universe runner and commit `2ccb2134...` |
| ATM IV | Mean of the selected call and put implied volatilities. The broad research selects an exact-prior-market-session option chain, nearest eligible 7–45 DTE expiry, and a common ATM strike using the frozen quality/ranking rules. | `broad_conflict_options_iv_screen_v0.py`, broad-universe runner |
| Missing bars | Missing `T0` gives `MISSING_T0_OPEN`; missing exact `T0-3m` gives `MISSING_EXACT_T0_MINUS_3_OPEN`; no interpolation or partial result occurs. | `derive_pre_move`, accepted ledger |
| Invalid data | A non-finite or non-positive `T0` open gives `INVALID_T0_OPEN`. The executable assumes the upstream frozen `P0` and `M_price` are valid. | `derive_pre_move` |
| Numeric behavior | Python/Pandas floating-point operations are used in the displayed order. The producer applies no explicit rounding. | `derive_pre_move` |
| Adjacent pre-volume context | Three complete OHLCV one-minute bars ending at `T0-1m` are required; a missing field makes those separate metrics unavailable. | `pre_admission_price_volume_v0/run_experiment.py` |

The stock artefacts are labelled historical-provider/EODHD OHLC and contain extended-hours data.
The exact points used by these examples happen to be during the signal session, but that does not
prove an IBKR `useRTH` setting. The split-alignment transform above is confirmed; the vendor's raw
corporate-action adjustment semantics are not.

## Authoritative PRE_MOVE_M reference rows

The four compact reference rows below were selected from the accepted frozen ledger, one per
cohort. Re-evaluating the recovered historical producer against each referenced Parquet source
matched the stored value within `4.5e-15`. These are valid golden cases for the recovered
`PRE_MOVE_M` function **when `P0` and `M_price` are supplied**. They are not goldens for an absent
PRE-level calculation or for IBKR/EODHD equivalence.

| Row/cohort | `as_of` (`T0`, UTC) | raw open `T0-3m` | raw open `T0` | `P0` | `M_price` | expected `PRE_MOVE_M` |
|---|---:|---:|---:|---:|---:|---:|
| `HUM\|2025-01-03\|6` / BROAD2025 | 2025-01-03 15:00 | 258.945 | 258.5 | 258.5 | 1.03439451809599 | 0.430203362658094 |
| `ASTS\|2025-01-02\|6` / ORIGINAL20_ASSESSMENT | 2025-01-02 15:00 | 21.93 | 22.05 | 22.049999 | 0.189214934206869 | 0.634199383155613 |
| `IREN\|2024-01-30\|6` / ORIGINAL20_DEVELOPMENT | 2024-01-30 15:00 | 4.4999 | 4.54 | 4.539999 | 0.0582285855127329 | 0.688665039933564 |
| `NBIS\|2025-01-02\|6` / UNSEEN49 | 2025-01-02 15:00 | 29.118 | 29.4 | 29.399999 | 0.249417548249557 | 1.13063412092402 |

The accepted ledger also contains two natural missing-data cases:

- `MPWR|2025-04-10|6`
- `NTRS|2025-05-22|28`

Both remain unavailable with `MISSING_EXACT_T0_MINUS_3_OPEN`; neither is interpolated or assigned a
PRE_MOVE value.

## Unresolved trading-critical contract fields

These fields cannot be established from the repository or recovered history:

- what “PRE levels” means in the Stage 4 architecture, including its output fields and whether
  `PRE_MOVE_M` is intended to be the Stage 4 output or a later feature;
- an authoritative bars-only PRE-level formula, calculation order, version, and golden result;
- the IBKR `whatToShow` value, `useRTH` value, duration/lookback, bar timestamp interpretation,
  and corporate-action/adjustment semantics equivalent to the accepted research;
- an authoritative IBKR method for reproducing the exact-prior-session ATM option chain, frozen
  option-quality/ranking behavior, and `M_price`;
- the live meaning and availability time of `P0`, including its causal relationship to `T0`;
- session/calendar completeness rules beyond the exact timestamps used by the recovered
  `PRE_MOVE_M` producer, including half sessions and any required historical session count;
- an accepted IBKR input/output parity fixture demonstrating that IBKR bars and options reproduce
  the frozen EODHD research value.

No source proves that `TRADES`, RTH-only data, a particular session count, or an arbitrary rolling
window is correct. None may be selected as a default.

## Minimum information required to unblock Stage 4

1. Identify or deliberately define the exact Stage 4 PRE-level output and formula, and state
   whether the recovered `PRE_MOVE_M` is that output or a Stage 5 feature.
2. Supply the canonical live definitions and causal timing of `P0` and `T0`.
3. Supply the authoritative IBKR request contract: bar size, `whatToShow`, RTH mode, timezone and
   timestamp convention, lookback/session rule, completeness rule, and adjustment behavior.
4. If `PRE_MOVE_M` is required in Stage 4, authorize the exact IBKR prior-session option-chain and
   ATM-pair acquisition contract that replaces the external/EODHD Group-O research producer.
5. Supply at least one accepted IBKR-backed input/output case, or explicitly authorize a parity
   exercise that defines how IBKR output is judged equivalent to the frozen research artefact.

Providing those items as a new approved specification would be a deliberate methodology decision,
not recovery of a contract currently present in this repository.
