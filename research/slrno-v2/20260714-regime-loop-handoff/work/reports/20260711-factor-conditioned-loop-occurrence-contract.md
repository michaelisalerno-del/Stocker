# Factor-conditioned loop-occurrence algorithm — frozen contract

Status: **frozen before model fitting**  
Scientific status: development research on 2024; 2025 and backward-2023 are already-opened demotion-only panels, never prospective validation

`research_only: true`  
`live_ordering_enabled: false`  
`order_placement: disabled`

## Question

Does the probability that a fixed loop occurs improve when the retained causal
last-three-state pattern is combined with B0, stress, entry clock, and five
causal price-context factors observed at the run-entry bar?

This is an occurrence-likelihood test. It does not test direction, return,
movement quality, P&L, economic edge, tradability, or deployment. Passing it
cannot promote a loop to good/high movement quality; that requires the separate
consequence contract, under which no portable good/high loop currently exists.

## Full causal population

The experiment keeps every run entry used by the frozen identity model:

| Period | Run entries | Compatible loop rows | Role |
|---|---:|---:|---|
| 2024 | 110,949 | 759,212 | causal development |
| 2025 | 109,658 | 711,877 | opened development, demotion only |
| backward-2023 | 107,159 | 709,455 | opened portability, demotion only |

The future-price-horizon anchor panels are forbidden because they exclude late
run entries and would change loop/session-end prevalence. Each run is joined
one-to-one to its provider bar by symbol, session date, and start timestamp.
No unmatched row may be dropped.

The provider bars are regular-session five-minute provider OHLCV. No volume
feature is used; the required label is `historical_volume_not_used`.

The raw 2024 provider files contain 5,539 regular-session placeholder rows with
all four OHLC fields null and no intersection with any frozen run-entry key.
The original pre-state lineage discarded these rows. This experiment performs
the same explicit pre-feature cleanup, hashes the discarded symbol/timestamp
keys, stops on any partial-null row, and still requires a one-to-one match for
all 110,949 run entries. No outcome-dependent row removal is allowed.
The retained 2024 causal factor table is additionally frozen by canonical SHA-256
`f07a1feae8aa4e61092131f659648eaaf39fdb8ce211d3e7f931b56abda1891a`.

## Target and factor timing

At the close of the current five-minute bar, the model emits overlapping
probabilities for all fixed cycles compatible with the current state. A label
is positive when the next two to four filtered destinations complete a valid
rotation of that cycle. Session end remains explicit destination 8.
Terminal run entries remain in model fitting, selection, pooled evaluation, and
ranking with zero loop labels; only their terminal-only diagnostic slice is
exempt from a positive-support floor.

Pattern inputs are previous-state-2, previous-state-1, current state, and the
fold-local retained history-path probability. The nine entry factors are:

- B0 numeric state and stress;
- entry-clock sine/cosine recomputed from the run start, including seconds;
- current-bar log return and range/open;
- rolling six-bar return sum;
- rolling twelve-bar mean absolute return;
- session-to-date return.

Only bars at or before the forecast origin may contribute. Stored run-end clock,
duration, end position/time, next state, future price, stock identity, and all
movement outcomes are excluded.

## Direct residual algorithm

The candidate is a direct compatible anchor-cycle residual head:

`logit(q) = logit(qhistory) + residual(pattern, cycle, route, entry factors)`

The offset coefficient is fixed at one. Factors are used once at entry and are
not repeated down hypothetical future transitions. The hierarchy contains a
global residual intercept, nineteen weighted sum-to-zero cycle contrasts,
twenty-four within-cycle route contrasts, and global, cycle, route, and
history-token factor interactions. Its exact widths are 44 for pattern-only,
2,812 for limited4, and 6,272 for full9. Cycle and route contrast weights come
only from the training prefix. History-token factor columns are sparse, not
centered, and unsupported tokens back off exactly to global/cycle/route
effects. Increasingly fine blocks receive stronger ridge penalties.

All nine factors use the prefix transform `(x - median) / median-centered RMS`;
missing B0 is first mapped to neutral zero exactly as in the retained lineage.
Four ridge values are selected causally from a frozen grid. Outer OOF months are
July–December 2024; every fit uses only earlier sessions. The literal inner
schedule begins in April. An outer month may inform later folds and the final
2024 development fit, but never its own prediction. Every outer fold uses a
newly fitted history kernel from its own training prefix. Using the final
full-2024 history model in OOF would leak outer-month outcomes and stops the run.

Matched controls receive the same folds, solver, grid, and structural capacity:

- `qpattern`: history offset plus pattern/cycle/route residuals, factors zero;
- `qlimited4`: the same head with B0/stress/clock interactions;
- `qfull9`: the same head with all nine factors;
- `qhistory`: the retained fold-local history path;
- the old limited chained context model is an additional required no-worse
  baseline, while matched `qlimited4` isolates factor value fairly.

Raw probabilities are primary. No post-hoc calibration is allowed.

## Required evidence

In 2024 OOF, and separately in each later period if scoring is authorized,
`qfull9` must beat all three primary baselines. Minimum pooled log-loss gains
are 1% versus history, 0.5% versus matched pattern-only, and 0.25% versus
matched limited4. Brier loss and five-session moving-block uncertainty must
also improve. Top-three recall must rise by 0.5 percentage points versus
history and 0.2 points versus each matched head without reducing precision.

Raw calibration diagnostics must have ECE no worse than the three primary baselines and maximum
supported-bin error at most 0.02. Improvements must survive every time slice,
every leave-one-stock-out deletion, at least 15/20 cycles, all supported current
states and transition lengths, inverse-compatible-cycle weighting, the
nonterminal subset, and the causal early-entry subset. A terminal/late-clock
effect cannot pass as general loop prediction.

The incremental full9-versus-limited4 alignment is challenged with 999 frozen
session-boundary circular shifts of the complete compatible-cycle residual
vector and Holm correction. The same shift is applied to every compatible
cycle at an anchor, preserving the top-three ranking null. Six primary proper-
loss uncertainty endpoints use a common five-session resample matrix and a
Bonferroni familywise one-sided bound. Exact zero-block embedding tests must
recover limited4, pattern-only, and history predictions.

## Scoring lock and interpretation

An independent implementation must reconstruct the complete 2024 fit, OOF
predictions, features, labels, tuning, metrics, gates, and falsification before
later scoring can be authorized. If any 2024 gate fails, the test stops and
2025/2023 are not opened by this experiment. If it passes, those later periods
can only preserve or demote the candidate.

The strongest possible result is
`development_candidate_retained_pending_prospective`. It would mean that entry
factors improve loop-occurrence probabilities on opened historical panels. It
would not establish good/high movement performance or prospective reliability.

The exact machine-readable contract is
`work/contracts/20260711-factor-conditioned-loop-occurrence-v1.json`.
