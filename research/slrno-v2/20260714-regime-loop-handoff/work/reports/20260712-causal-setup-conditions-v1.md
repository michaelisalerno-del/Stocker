# Causal setup conditions V1

Date: 2026-07-12

Decision: `no_setup_condition_retained`

Scientific status: July-December 2024 monthly-OOF setup-pattern research. The underlying state definition was fitted on full 2024, so this is internal development evidence, not prospective validation.

Safety:

- `research_only: true`
- `live_ordering_enabled: false`
- `order_placement: disabled`
- no live, demo, paper, broker, position, deployment, or strategy-promotion path exists
- provider-volume label: `historical_volume_not_used`
- no setup or loop was promoted

## Question

Can causal setup conditions supply the directional and entry information missing from the retained state-history movement forecast?

The experiment followed the bounded bar-pattern workflow. It froze five falsifiable hypotheses before scoring any new setup return:

1. replace intrabar breakout touches with a completed close outside the anchor range;
2. require the monthly-OOF state-history range forecast to exceed its frozen 2024 P75;
3. require a strong confirmation bar;
4. require prior range compression;
5. require alignment with the trailing six-bar trend.

The data comprised 34,169 exact OOF anchors, twenty-two stocks, and 128 sessions. Only regular-session five-minute provider OHLC was used.

## Setup definitions

Close confirmation inspected the next three completed bars after an anchor:

- close above anchor high: long confirmation;
- close below anchor low: short confirmation;
- entry at the following bar open;
- exit at the anchor-plus-6/12/24-bar close;
- no stop or take-profit;
- one non-overlapping armed window per stock.

The frozen movement gate used the monthly-OOF history-layer range forecast and the already-frozen full-2024 raw-history prediction P75:

- 6 bars: 210.32 bps;
- 12 bars: 283.32 bps;
- 24 bars: 372.92 bps.

Strong close required:

- absolute body at least 50% of confirmation-bar range; and
- close in the outer 25% of that bar in the confirmed direction.

Compression required the mean range percentage over the anchor and previous five bars to be no more than 75% of the corresponding 24-bar mean. Trend alignment required the confirmed direction to match the sign of the trailing six-bar close return.

All primary results use five bps per side.

## Main result

| Setup | Horizon | Trades | Mean net trade | Annualized return | Maximum drawdown |
| --- | ---: | ---: | ---: | ---: | ---: |
| Intrabar OCO baseline | 6 | 13,672 | -1.96 bps | -22.66% | -21.53% |
| Intrabar OCO baseline | 12 | 9,099 | -1.15 bps | -9.82% | -23.11% |
| Intrabar OCO baseline | 24 | 6,025 | +2.72 bps | +12.72% | -16.06% |
| Close-confirmed, ungated | 6 | 12,193 | -7.99 bps | -58.45% | -35.99% |
| Close-confirmed, ungated | 12 | 7,834 | -6.56 bps | -37.46% | -24.98% |
| Close-confirmed, ungated | 24 | 5,072 | -4.69 bps | -21.07% | -20.29% |
| History movement gate | 6 | 4,573 | -7.64 bps | -27.34% | -16.23% |
| History movement gate | 12 | 3,474 | -3.82 bps | -12.05% | -13.72% |
| History movement gate | 24 | 2,626 | +0.07 bps | -1.74% | -12.01% |
| **History gate + strong close** | **6** | **2,793** | **-4.01 bps** | **-10.07%** | **-7.99%** |
| **History gate + strong close** | **12** | **2,135** | **+1.55 bps** | **+2.04%** | **-8.69%** |
| **History gate + strong close** | **24** | **1,620** | **+6.85 bps** | **+9.39%** | **-8.13%** |
| History gate + compression | 6 | 406 | -31.85 bps | -10.95% | -6.00% |
| History gate + compression | 12 | 403 | -35.20 bps | -11.96% | -6.53% |
| History gate + compression | 24 | 453 | -47.51 bps | -17.61% | -10.11% |
| History gate + trend alignment | 6 | 1,699 | -12.39 bps | -17.19% | -9.35% |
| History gate + trend alignment | 12 | 1,027 | -20.77 bps | -17.45% | -9.60% |
| History gate + trend alignment | 24 | 580 | -14.55 bps | -7.33% | -4.14% |

## What the hypotheses showed

### H1: close confirmation — rejected

A close outside the anchor range did not solve false-breakout behaviour. It was worse than the intrabar OCO baseline at every horizon. At six bars the paired daily interval was entirely adverse; at 12 and 24 bars it still did not demonstrate improvement.

Waiting for a close consumed part of the move but did not prevent enough reversal afterward.

### H2: movement gate — useful filter, not a profitable setup

The state-history range gate improved close confirmation at all three horizons. Every paired moving-block interval versus ungated close confirmation was beneficial:

- 6 bars: daily advantage interval `[+0.1550%, +0.2663%]`;
- 12 bars: `[+0.0524%, +0.2114%]`;
- 24 bars: `[+0.0042%, +0.1642%]`.

This is strong evidence that the regime/history forecast filters setup quality. But the gated setup remained absolutely negative at 6 and 12 bars and essentially flat at 24. Its absolute bootstrap intervals did not establish positive returns.

### H3: strong confirmation close — promising but unretained

Strong closes materially improved the movement-gated baseline:

- 6-bar paired improvement was significant, although absolute return remained negative;
- 12-bar paired improvement was barely significant and the average became positive;
- 24-bar average became positive, but the paired interval crossed zero.

The absolute daily-return interval crossed zero at every horizon. Monthly persistence was weak:

- 6 bars: 1 of 6 months positive;
- 12 bars: 3 of 6;
- 24 bars: 4 of 6.

Leave-one-stock-out cumulative return was positive for 0 of 22 deletions at 6 bars, 18 of 22 at 12 bars, and all 22 at 24 bars. This is a useful clue, not dictionary-wide robustness.

Strong-close results were also friction-sensitive. At two bps per side all three annualized cells were positive; at five bps the six-bar cell failed.

### H4: compression — rejected strongly

The fixed compression definition produced fewer than 500 trades per horizon and substantially negative average returns. It failed support and performance gates. The data do not support the simple idea that a 6-versus-24-bar range ratio below 0.75 identifies a favorable continuation setup here.

### H5: trend alignment — rejected

Requiring the confirmed break to agree with trailing six-bar direction made performance worse. Every mean trade and annualized-return cell was negative. This indicates that the close-confirmed move frequently exhausts rather than continues, or that the six-bar trend measure is redundant and late.

The opposite countertrend rule was not tested after seeing this result; flipping the condition now would be post-score fitting.

## Session dependency diagnostic

The strong-close effect was concentrated in the earliest clock quartile. At five bps per side, mean trade return for that slice was:

- 6 bars: +0.89 bps;
- 12 bars: +11.79 bps;
- 24 bars: +25.42 bps.

Every represented later quartile was negative. This slice was inspected after scoring and had no frozen retention gate. It cannot be promoted from this experiment.

It does define the clearest next hypothesis: state-history P75 movement forecast plus strong close during the first session quartile, evaluated only under a new contract on genuinely unseen sessions. It should not be retested or tuned on these same 2024 outcomes.

## Decision

No setup passed the full support, absolute-return, bootstrap, month, stock-deletion, and paired-improvement gates.

The defensible findings are narrower:

- regime/history movement forecasts consistently improve setup selection;
- close confirmation by itself is not enough;
- strong confirmation bars contain additional information, especially at 12/24 bars;
- the apparent strong-close effect is time-dependent and not yet stable;
- the tested compression and trend-alignment definitions should be rejected;
- no strategy or economic edge is established.

## Skill-workflow impact

The bar-pattern research workflow kept the experiment to five predeclared hypotheses, required exact OHLC and volume/proxy labels, forced causal next-bar entries after confirmation, and treated every result as pattern intelligence rather than strategy promotion. That prevented post-result parameter flips from being counted as evidence.

## Integrity and reproducibility

- contract SHA-256: `5599d49838274d3aad944ecdd51b83b191e6a903fa97e01e32b3b55ddd91ab6d`
- corrected pre-score runner SHA-256: `be19c2e8c0473a24115572fd512b3124138e03a378eaf0f529eae90082458254`
- independent auditor SHA-256: `f6326fb7cd241a9613a5024b71b3266f059385e9848204fe1536ddab106c6261`
- independent audit-result SHA-256: `aaf261fefb943cfbb8a2259d4e1045979699d4c3af0fdcd2191168e0381ed4ea`
- independent audit: 11/11 checks passed
- all 110,359 accepted setup rows reconstructed
- features, confirmations, entries, costs, overlap, returns, monthly slices, clock slices, stock deletions, bootstraps, and decision reconstructed
- exact setup feature and daily-return reconstruction error: `0.0`
- full workspace research suite: 294 tests passed

Artifact root: `/private/tmp/stocker_causal_setup_conditions_v1_20260712`

The artifact root is ephemeral and should be archived before reboot if exact replay without recomputation is required.
