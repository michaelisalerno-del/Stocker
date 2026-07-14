# Causal conditions before frozen regime-loop families V1

Date: 2026-07-13

`research_only: true`  
`live_ordering_enabled: false`  
`order_placement: disabled`

## Subsequent exact-identity correction

The follow-up identity audit found that successful parent cycle identity is
already determined exactly by the frozen orientation key. The cycle mixtures
reported below describe the same admission occurring across several different
`A->B` orientations; they are not evidence that parent identity remains
ambiguous after an orientation is fixed. The admissions help predict parent
completion and quick-versus-persistent route. See
`20260713-admission-exact-cycle-identity-v1.md` for the structural audit and the
remaining ambiguous morph/child branches.

## Decision

Conditions before a loop do matter, but not equally for every loop family.

The strongest repeatable structure is between **quick parent returns** and **persistent parent returns**:

- small or near-VWAP transition bars favor quick returns;
- large or far-from-VWAP opening transitions favor persistent returns;
- one ordered three-bar admission favors quick returns;
- a different ordered three-bar admission favors persistent returns.

These are probability shifts, not deterministic causes. More than one distinct admission can lead to the same loop family, and each admission still maps to several frozen cycle identities.

Morph and child antecedents remain unresolved. Each retained one weak broad map after strict context adjustment, but neither had a confidence interval above zero in both outside years. No ordered child sequence survived.

The large “put every lagged condition into one matrix” idea failed. Adding all exact `t-1`, `t-2`, and `t-3` values to the full model did not improve outside-year log loss for any family and robustly degraded several comparisons. The useful result is therefore a **small set of admissions**, not a larger unrestricted dictionary.

No P&L, execution, or tradability calculation was performed.

## Question tested

At the first completed bar of a frozen regime transition, do causal bar, direction, provider-volume, VWAP, session, market, and regime-path conditions identify which kind of frozen loop is more likely to complete?

Four target families were kept separate:

| Target | Opportunity | Positive outcome |
|---|---|---|
| `parent_quick` | observed `A -> B` | next transition is `B -> A`, with B lasting one bar |
| `parent_persistent` | observed `A -> B` | next transition is `B -> A`, with B lasting at least two bars |
| `morph` | observed `A -> B` | next two transitions complete a frozen morph |
| `child` | observed parent closure `A -> B -> A` | next two transitions complete a frozen child |

Run duration was used only to define the quick/persistent outcome. It was never an input feature.

## Data and causal clocks

- Frozen EODHD-provider five-minute regular-session OHLCV tape.
- Twenty US stocks, 2023 through 2025.
- Frozen eight-state causal semi-Markov detector and frozen twenty-cycle dictionary.
- 600,419 target rows.
- Provider volume is labelled `historical_volume`; no quote count, tick count, spread, order book, or news feed was available.

Two information clocks were evaluated separately:

1. **Strict pre-transition:** completed bars ending at `t-1`, including a trailing-three-bar summary.
2. **Onset:** the completed transition bar at `t`, before any future transition or loop completion.

The ordered-history extension used exact `t-3`, `t-2`, and `t-1` conditions. On a contiguous five-minute segment these correspond to approximately 15, 10, and 5 minutes before the transition bar.

MFE and MAE were not used as preconditions because they are future path outcomes. Causal range, range expansion, body size, and bar movement supplied the available ATR-like information.

## Validation design

- H1 2024: threshold fitting and candidate discovery.
- Q3 2024: independent confirmation and selection.
- Q4 2024: descriptive only.
- 2023 and 2025: unchanged outside-year tests.
- Numeric low/high thresholds were frozen at H1 2024 33rd/67th percentiles.
- Broad rules contained exactly two tokens from different condition families.
- Ordered rules contained two tokens at different lags.
- At most three broad maps and two ordered maps were selected per target.
- Outside-year effects used five-session block bootstrap with 5,000 draws.

The first broad pass adjusted for regime orientation. A frozen validation pass then re-scored the same selected maps within **regime orientation × stock × session bucket**. No map was changed or replaced during that audit. All retained conclusions below use this stricter context-adjusted result.

Baseline reconstruction matched the frozen opportunity and success counts exactly in all three years.

## Strong broad condition maps

Effects below are context-adjusted probability differences in percentage points. Intervals are two-sided 95% five-session block-bootstrap intervals. Raw occurrence rates are shown only to communicate scale; for session-conditioned rules, whole-day raw baselines are not the correct causal comparison.

| Loop family | Conditions known at `t` | 2023 adjusted effect | 2025 adjusted effect | Raw occurrence rate, 2023 / 2025 |
|---|---|---:|---:|---:|
| Quick parent | transition body no more than 11.89 bps and onset direction down | +5.67 pp `[+4.37, +6.95]` | +6.87 pp `[+5.14, +8.58]` | 23.77% / 22.86% |
| Quick parent | first 30 minutes and transition close within 55.78 bps of session VWAP | +9.13 pp `[+5.39, +13.39]` | +11.91 pp `[+8.82, +15.00]` | 17.88% / 19.03% |
| Persistent parent | first 30 minutes and transition close at least 133.96 bps from session VWAP | +8.26 pp `[+3.23, +13.40]` | +5.45 pp `[+0.07, +10.90]` | 45.44% / 47.18% |
| Persistent parent | first 30 minutes and transition body at least 36.04 bps | +5.69 pp `[+1.65, +10.90]` | +5.34 pp `[+0.41, +10.77]` | 36.68% / 39.94% |

Outside-year whole-day baselines were 18.68% and 17.57% for quick parents, and 24.12% and 24.99% for persistent parents.

The opening-session maps form a clean mirror: **near VWAP favors quick closure; far from VWAP favors persistence**. Body size shows the same division: small transition bodies favor quick closure, while large opening bodies favor persistence.

One additional quick-parent map—downward pre-direction in the closing 90 minutes—was positive in both outside years but not strong in 2025. It remains descriptive, not retained as firm evidence.

## Strong ordered admissions

Only two ordered `t-3` to `t-1` sequences had confidence intervals entirely above zero in both outside years:

| Loop family | Ordered conditions before `t` | 2023 adjusted effect | 2025 adjusted effect | Raw occurrence rate, 2023 / 2025 |
|---|---|---:|---:|---:|
| Quick parent | absolute move at `t-1` at least 29.49 bps, and regime at `t-2` differs from the regime at `t-1` | +3.08 pp `[+2.16, +3.98]` | +3.91 pp `[+2.83, +5.02]` | 20.04% / 19.68% |
| Persistent parent | absolute move at `t-3` at least 34.23 bps, and `t-1` at least 129.49 bps from session VWAP | +1.68 pp `[+0.66, +2.66]` | +2.22 pp `[+1.12, +3.27]` | 25.10% / 28.15% |

The quick admission means the pre-transition regime itself began only one bar earlier, after a relatively large move. This is a rapid state-churn or burst signature and is consistent with a quick return.

The persistent admission instead describes displacement established earlier and still far from VWAP immediately before the transition. It is consistent with a move that has retained location rather than a one-bar detector flicker.

Other ordered findings were weaker:

- a second quick sequence, two persistent sequences in total, and two morph sequences were directionally portable;
- only one quick and one persistent sequence were statistically strong;
- both selected child sequences reversed sign outside the selection period, leaving zero portable ordered child maps.

## Mapping conditions back to frozen cycle identities

The conditions do not select a single cycle identity. They change the mixture of identities.

- Small downward quick-transition bodies mapped mainly to cycles `05`, `01`, and `09`. Those three represented 82.7% of selected quick successes in 2023 and 72.5% in 2025.
- Near-VWAP opening quick transitions formed a different admission. Cycle `06` was largest at 36.8% in 2023 and 43.1% in 2025.
- Far-from-VWAP opening persistent transitions mapped almost entirely to cycles `07`, `06`, and `13`: 97.9% in 2023 and 99.0% in 2025.
- Large-body opening persistent transitions were led by cycle `06` at approximately 50% in both years, followed by `07` and `13`.
- The strong ordered persistent admission enriched cycle `07` by about 1.9–2.0 times its family baseline share and cycle `13` by about 2.8–3.4 times in both outside years.
- The strong ordered quick admission repeatedly enriched cycles `08`, `09`, and `10`, but did not collapse to one dominant identity.

This supports a hierarchical interpretation:

1. regime orientation defines what loops are structurally eligible;
2. pre-transition path and onset geometry shift quick-versus-persistent family probability;
3. the same admission still branches among several frozen identities.

There is also evidence for multiple admissions to the same family. The two strong quick broad maps shared no successful outside-year events. The two strong persistent broad maps had success-event Jaccard 0.352. For the two ordered quick maps the Jaccard was 0.048; for the persistent maps it was 0.003. They are not merely duplicate descriptions of the same cases.

## Morph and child result

The strict audit retained only weak evidence:

- Morph: downward onset plus an absolute transition-leg return no more than 14.91 bps. The adjusted effect was +3.40 pp in 2023 but only +0.37 pp in 2025; the 2025 interval crossed zero.
- Child: low trailing-three body and range. The adjusted effect was +0.14 pp in 2023 and +2.04 pp in 2025; both intervals crossed zero.

The morph cases continued to map primarily to cycles `18` and `19`. The child cases mapped mainly to `12` and `17`, but their mixture drifted substantially between years.

This does not solve morph or child prediction. It is consistent with the prior detector-robustness finding that child labels are especially detector-sensitive.

## Predictive ablations

The full-model checks prevent over-interpreting selected rule lifts.

- Strictly pre-transition summary features improved log loss at the point estimate in both outside years for persistent parents and children, but no family had a robust improvement in both years.
- Onset features robustly improved quick-parent log loss in both outside years: relative improvements were 0.516% in 2023 and 1.184% in 2025.
- Morph onset evidence was inconsistent across years.
- Adding all exact ordered-lag values on top of the trailing summary failed for every family. In 2023 it robustly degraded log loss for quick parent, persistent parent, morph, and child. In 2025 it robustly degraded persistent-parent log loss and did not robustly improve any family.

Therefore the data support a few sparse admissions, particularly at the transition bar, but do not support an unrestricted high-dimensional history matrix.

## Interpretation

The useful working model is not “regime + direction mechanically produces loop N.” It is:

> A frozen regime transition creates an eligible set of loops. Recent state churn, displacement, VWAP location, session, and transition-bar geometry alter the conditional odds of quick versus persistent closure. Several distinct admissions can lead to the same family, and exact identity remains probabilistic.

This is closer to the proposed admission idea, but it is not a 90% next-loop classifier. The strongest raw persistent-parent filters reached approximately 37–47% occurrence, while the strong quick filters were approximately 18–24%. Context-adjusted lift is useful evidence, but it is not precision and says nothing by itself about profit.

## Recommended next research test

Do not expand the dictionary with every lagged condition.

Freeze the six strong admissions above as sparse flags and test a hierarchical forecast:

1. existing regime forecast;
2. quick-versus-persistent parent route using the sparse admissions;
3. conditional cycle-identity probabilities within the admitted family;
4. abstention when no admission is present;
5. prospective logging on genuinely new sessions before outcomes are attached.

Morph and child should remain separate research problems. They should not borrow confidence from the stronger parent result.

## Integrity and artifacts

No app/runtime source was changed. No orders, deployment, or P&L calculation occurred.

Research scripts:

- `/private/tmp/run_preloop_condition_map_test.py`
- `/private/tmp/run_ordered_preloop_sequence_test.py`
- `/private/tmp/run_preloop_context_audit.py`

Artifact roots:

- `/private/tmp/stocker_preloop_condition_maps_20260713`
- `/private/tmp/stocker_ordered_preloop_sequences_20260713`
- `/private/tmp/stocker_preloop_condition_maps_context_audit_20260713`

Exact rerun summary hashes:

- broad condition maps: `e1c08d2d48530a92d075411ae68dd315f652ce65008ec06af73cd23efaa1a89d`
- ordered sequence test: `80ad44f47535728ceab292d132303287fcb4eb75c74141fea7e64bfba365f0eb`
- frozen context audit: `e6354470910e318a8d634080de169c007168c1f18ea59cb1c02f254366352d43`

The `/private/tmp` artifacts are ephemeral and should be archived before a reboot if exact ledgers are required without recomputation.
