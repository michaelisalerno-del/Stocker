# Per-loop movement-quality contract

## Scope and interpretation

This contract was frozen before reading any 2025 or backward-2023 row-level
movement outcome for this experiment. It adds a separate movement-quality
layer to the retained loop-identity forecaster. It does not refit the frozen
eight-state detector, change the twenty cycles, change the retained structural
loop probabilities, or modify the existing prospective movement shadow
contract or ledger.

`research_only: true`, `live_ordering_enabled: false`, and
`order_placement: disabled`. The labels in this experiment mean
`high_movement_quality`, `good_movement_quality`, or `unqualified`. They do not
mean high trading performance, direction, profitability, economic edge, or
tradability. No signed-return, long/short, P&L, cost, spread, slippage,
position, broker, order, strategy, or deployment field is permitted.

2025 remains a development period and backward-2023 remains a future-fitted
portability period. They may demote a 2024 provisional grade, but neither can
establish prospective validity or promote a cycle above its frozen 2024
provisional grade.

## Separate structural and quality probabilities

For compatible cycle `c` at causal run-entry anchor `i`, retain the existing
structural probability:

`s(i,c) = P(loop c's required state path occurs | causal state history)`.

Fit a separate conditional movement distribution only on 2024 examples where
that loop path actually occurred:

`q(i,c) = P(movement class | loop c occurs, causal anchor information)`.

Future loop identity is a 2024 training label, never an inference feature. If
an anchor satisfies two overlapping loop labels, each positive anchor-cycle
row receives weight one-half; otherwise it receives weight one. More
generally, the weight is the reciprocal of the number of positive fixed-cycle
labels at the anchor.

The combined outputs are exact chain-rule probabilities:

- `j75(i,c) = s(i,c) * q75(i,c)`;
- `j90(i,c) = s(i,c) * q90(i,c)`.

Their targets are respectively `loop_occurs * exceeds_P75` and
`loop_occurs * exceeds_P90`. This is not an independence assumption because
`q` is explicitly conditional on the loop occurring. Cycles overlap, so joint
probabilities are never summed into a mutually exclusive cycle distribution.
Every result must report `s`, `q75`, `q90`, `j75`, and `j90` separately.

## Frozen movement targets

Anchors are exact regular-session causal run entries with zero-based New York
bar ordinal at most 53 and exact five-minute future support through 24 bars.
The completed anchor bar is available; outcomes begin with the following bar.

- Absolute return is `abs(10000 * log(close[t+h] / close[t]))`.
- Future range is
  `10000 * (max(high[t+1:t+h]) - min(low[t+1:t+h])) / close[t]`.

The following global thresholds use the 2024 anchor panel only, pandas linear
quantile interpolation, and a strict greater-than event definition:

| Target | Horizon | 2024 P75 (bps) | 2024 P90 (bps) |
| --- | ---: | ---: | ---: |
| Absolute return | 6 | 116.959429 | 200.729980 |
| Absolute return | 12 | 158.554774 | 269.304369 |
| Absolute return | 24 | 212.952622 | 354.915227 |
| Future range | 6 | 217.391123 | 333.789058 |
| Future range | 12 | 295.503419 | 443.438949 |
| Future range | 24 | 396.872721 | 577.227279 |

For each target and horizon, model one ordered three-class outcome:

1. class 0: outcome at or below P75;
2. class 1: outcome above P75 and at or below P90;
3. class 2: outcome above P90.

Then `q75 = P(class 1) + P(class 2)` and `q90 = P(class 2)`, guaranteeing
`0 <= q90 <= q75 <= 1` without independently fitted probabilities crossing.

## Frozen conditional models

Both models are multinomial logistic regressions with `C=0.2`, `lbfgs`, at
most 1,000 iterations, and seed 20260710. They use identical 2024-only median
imputation and `StandardScaler(with_mean=False)` for the shared controls.

The shared causal controls are current-state one-hot; B0 numeric state and
stress; causal entry-clock sine/cosine; current-bar log return and range;
trailing six-bar return; trailing twelve-bar mean absolute return; and
session-to-date return. Stock identity, future state, realized duration,
future price, and stored run-end clock are excluded.

- `qcontext` uses only the shared controls. It estimates movement quality
  conditional on some frozen loop occurring, without knowing which cycle.
- `qcycle` adds a twenty-cycle one-hot block scaled by 1, a
  cycle-by-current-state block scaled by 0.5, and a
  cycle-by-648-history-token block scaled by 0.25.

Under common L2 regularization those blocks have effective relative penalties
of 1, 4, and 16. Weak or unseen fine interactions therefore shrink toward the
cycle main effect and shared context rather than producing unsmoothed
per-cycle rates. No feature block, scale, regularization value, or control may
change after scoring begins.

## Causal 2024 OOF calibration and provisional grades

For each month July through December 2024, fit only on strictly earlier 2024
months and predict that month. All fitting and OOF scoring use realized-loop
positive rows with inverse-overlap weights.

For each model, target, and horizon, temperature-calibrate the three class
probabilities with the fixed grid `{0.75, 1, 1.25, 1.5, 2}`. Select the
temperature with the lowest weighted July–December OOF multinomial log loss;
break ties toward 1, then toward the lower value. Refit the unchanged model on
all eligible 2024 positive rows and attach the selected OOF temperature. This
fit, its thresholds, provisional cycle grades, hashes, and an independent
pre-score audit must be sealed before a 2025 or 2023 row-level outcome is
opened.

Full-2024 fit eligibility requires, per cycle, at least 5,000 compatible rows,
500 realized loops, eighteen stocks, all four quarters, and fifty realized
loops per quarter.

The July–December OOF provisional-grade cohort requires at least 250 realized
loops, fifteen stocks, both Q3 and Q4 with at least 75 realized loops each, and
the following endpoint support:

- good: at least fifty P75 positives and fifty negatives;
- high: additionally at least twenty-five P90 positives and fifty negatives.

Support failure makes the cycle unqualified. Backoff cannot manufacture a
grade where the cycle itself lacks evidence.

## Per-horizon quality grades

Grades are calculated separately at 6, 12, and 24 bars. Both absolute return
and future range must satisfy every applicable rule; a one-target pass cannot
qualify.

### Good movement quality

For each target's P75 event:

- observed conditional exceedance rate and mean calibrated `qcycle75` are at
  least 0.30;
- the observed rate is at least 1.10 times mean `qcontext75` on the same
  realized-loop rows;
- the five-session moving-block 95% lower bound of
  `event - qcontext75` is above zero;
- conditional `qcycle` log loss improves by at least 0.5% versus `qcontext`,
  Brier loss is lower, and both daily loss-difference upper bounds are below
  zero;
- joint `s*qcycle75` log loss improves by at least 0.25% versus
  `s*qcontext75`, Brier is lower, and both daily upper bounds are below zero;
- conditional and joint log loss and Brier are lower in every required
  quarter and every leave-one-stock-out deletion;
- fixed-ten-bin ECE is no worse than baseline, with maximum supported-bin
  error no more than 0.02 worse conditionally and 0.01 worse jointly.

OOF supported calibration bins require fifty conditional or 250 joint rows.

### High movement quality

High must first pass every good gate. For both targets:

- P75 observed rate and mean calibrated `qcycle75` are at least 0.35;
- P90 observed rate and mean calibrated `qcycle90` are at least 0.15;
- P90 observed rate is at least 1.20 times mean `qcontext90`;
- every conditional and joint proper-loss, interval, quarter,
  stock-deletion, residual-lift, and calibration gate also passes for P90.

A cycle's provisional global grade is high only if all three horizons are
high. It is good if all horizons are at least good and at least one is good.
If any horizon is unqualified, the global cycle grade is unqualified.

## Full-period demotion gates

After the contract, fit, hashes, and independent pre-score audit are sealed,
apply the unchanged models separately to full-period 2025 development and
backward-2023 portability data.

Each cycle and period requires at least 5,000 compatible rows, 500 realized
loops, eighteen stocks, all four quarters, and fifty realized loops per
quarter. A good test requires at least 100 P75 positives and 100 negatives per
target/horizon. A high test additionally requires at least fifty P90 positives
and 100 negatives.

Recalculate every rate, lift, conditional loss, joint loss, moving-block
interval, ten-bin calibration result, all-four-quarter slice, and every
leave-one-stock-out deletion on each full period. Scoring-period supported
bins require 100 conditional or 500 joint rows.

A qualified cycle must also retain structural reliability in each period: its
frozen history-path probability must have lower log loss and Brier than first
order, ECE no worse, and maximum supported-bin error no more than 0.01 above
first order.

The final grade is the minimum of the frozen July–December 2024 OOF
provisional grade, the full-2025 grade, and the full-2023 grade. A provisional
high cycle may demote to good; a failed good cycle becomes unqualified. No
2025/2023 result may promote a cycle, substitute another cycle, alter a
threshold, relax a gate, or change a feature or calibration choice.

## Prospective separation and decision boundary

Even a cross-period pass is development/backward-portability evidence only.
This quality layer cannot alter the already sealed prospective movement
shadow predictions, model hashes, ledger, cohort, or gates. A separate future
contract, frozen after this experiment, is required before any prospective
per-loop quality claim.

At inference, structural ranking remains the retained history model. The
quality layer may mark a structurally predicted loop as high, good, or
unqualified, but may not refit or silently reorder `s(i,c)`. Structural
probability and conditional quality must both be shown; neither is allowed to
stand in for the other.

No result from this contract permits a directional, long/short, trading,
economic-edge, or deployment claim.
