# Frozen regime/loop P&L sanity test V1

Date: 2026-07-12

Decision: `pnl_translation_not_supported`

Scientific status: post-inspection 2025 development and backward-2023 portability P&L diagnostic. This is not prospective validation or evidence of an economic edge.

Safety:

- `research_only: true`
- `live_ordering_enabled: false`
- `order_placement: disabled`
- no live, paper, demo, broker, order-submission, position-management, or deployment path exists
- direct provider-volume label: `historical_volume_not_used`
- no strategy or loop was promoted

## Question

Can the retained movement/range information be converted into hypothetical stock-only P&L?

The retained model does not predict direction. The test therefore separated:

1. direct long/short use of the already-rejected direction probability, as a falsification;
2. a causal breakout translation for the movement signal.

For the breakout, a forecast was formed after the completed anchor bar. Beginning with the next five-minute bar, the completed anchor high and low acted as upper and lower triggers. The first unambiguous break determined long or short direction. Entry gaps were filled adversely at the triggering-bar open, and the trade exited at the 6-, 12-, or 24-bar close.

Only anchors whose frozen predicted future range exceeded the representation's 2024 prediction P75 were admitted to the gated variants. Thresholds, overlap rules, costs, portfolio construction, and gates were frozen before scoring P&L.

## Execution and portfolio assumptions

- exact regular-session provider OHLC;
- common frozen universe of twenty stocks in both periods;
- one non-overlapping signal window per stock within each strategy/horizon;
- twenty equal capital sleeves, inactive sleeves held as cash;
- 250 zero-filled session dates in each period;
- cash-consistent simple long/short returns;
- exact sequential compounding within each stock sleeve;
- costs of 0, 1, 2, 5, and 10 bps per side;
- primary hurdle: 5 bps per side, or 10 bps round trip.

The provider bars contain no bid/ask quotes. Five bps is therefore a sensitivity assumption, not a measured execution cost.

## Primary loop-gated breakout result

At the primary 5-bps-per-side assumption:

| Period | Horizon | Trades | Mean net trade | Cumulative return | Annualized return | Maximum drawdown |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2025 development | 6 bars | 10,811 | -5.77 bps | -27.90% | -28.09% | -29.28% |
| 2025 development | 12 bars | 8,554 | +0.54 bps | +0.17% | +0.17% | -20.34% |
| 2025 development | 24 bars | 6,900 | +4.08 bps | +11.40% | +11.50% | -25.40% |
| backward-2023 | 6 bars | 10,203 | -5.51 bps | -25.39% | -25.57% | -26.25% |
| backward-2023 | 12 bars | 8,071 | -5.40 bps | -20.80% | -20.95% | -23.37% |
| backward-2023 | 24 bars | 6,307 | +0.16 bps | -1.92% | -1.94% | -15.00% |

Only two of six annualized-return cells were positive. Three of six mean-trade cells were positive. No absolute daily-return bootstrap lower bound exceeded zero.

Only 7 of 24 period/horizon/quarter cells were positive. Only 34 of 120 leave-one-stock-out cells were positive, concentrated in 2025 at the longer horizons.

Every predeclared retention gate failed.

## Gross opportunity exists, but it is friction-sensitive

Before costs, the same loop-gated breakout was positive in all six cells:

| Period | 6 bars | 12 bars | 24 bars |
| --- | ---: | ---: | ---: |
| 2025 gross annualized return | +24.00% | +54.18% | +57.87% |
| backward-2023 gross annualized return | +24.41% | +18.75% | +34.75% |

This is not a net edge. The mean-gross-return break-even cost per side was small and unstable:

| Period | 6 bars | 12 bars | 24 bars |
| --- | ---: | ---: | ---: |
| 2025 | 2.12 bps | 5.27 bps | 7.04 bps |
| backward-2023 | 2.25 bps | 2.30 bps | 5.08 bps |

At two bps per side, five cells remained positive, but 2025 at six bars was already slightly negative. At five bps, the cross-period result failed.

The proper conclusion is that predicted high-movement conditions contain gross price excursion, but this fixed breakout captures only a thin and execution-sensitive portion of it.

## Does the loop score itself add P&L value?

The loop gate did improve on trading every breakout:

- its paired daily advantage over ungated breakout was statistically positive at 6 and 12 bars in both periods;
- the 24-bar paired intervals crossed zero.

This means the movement forecast filters out many poor breakout conditions.

However, the loop-gated version did not show a reliable advantage over either:

- the state/context movement gate; or
- the raw state-history movement gate.

Every required paired interval versus state/context and raw history failed the all-cell gate. Several longer-horizon state/context cells were better than the loop version. Consequently, the result cannot be attributed specifically to loop probabilities.

This agrees with the regime-utility ablation: regime state and recent state history carry the main movement information, while loop probabilities are primarily a compact representation of that history.

## Directional P&L falsification

Directly converting the loop direction probability into long/short positions failed decisively at five bps per side.

Across both periods and all horizons:

- mean net trade return was negative in every cell;
- annualized return was negative in every cell;
- movement gating reduced losses but did not make the directional model reliable.

For example, the loop-direction all-anchor annualized returns ranged from -44.75% to -75.35% in 2025 and from -48.21% to -76.33% in backward-2023. This confirms that the movement result must not be described as directional prediction.

## Execution caveats make the rejection stronger

The simulation still contains favorable assumptions:

- approximately 2.9%-5.3% of primary signals whose first event bar touched both triggers were canceled as intrabar-path ambiguous; a real armed OCO order would have filled one side first;
- stop fills inside five-minute bars occur exactly at the trigger unless the bar opens through it;
- there is no measured bid/ask spread, additional slippage, latency, or market impact;
- intraday short locate, borrow, and short-sale-restriction constraints are absent;
- provider OHLC is used exactly as stored without an independent corporate-action reconstruction.

Because the primary candidate already fails under these favorable assumptions, adding more realistic execution friction would not rescue it.

## Decision

The system has measurable movement information, and that information can identify conditions with greater gross breakout opportunity. It is not presently a complete P&L mechanism.

The missing component remains directional/execution conversion:

- direct direction is not predictable enough;
- the fixed stock breakout is whipsaw- and friction-sensitive;
- loop scores do not outperform state or raw history reliably;
- 2025 and backward-2023 cannot establish future profitability anyway.

No P&L hypothesis is retained. No economic-edge, tradability, or strategy claim is permitted.

The scientifically sound next step is not to tune breakout distances, exits, costs, or thresholds on these opened periods. Continue the frozen prospective movement shadow. Any later economic test should use a new predeclared rule on genuinely unseen sessions and materially better execution data—quotes or intrabar ordering for stock breakouts, or options and implied-volatility data for a genuinely direction-neutral movement trade.

## Integrity and reproducibility

- amended common-universe contract SHA-256: `34ca60a11b4306c3d7282566c626a3848a7b4c55fda5a5fac2d09bf820b57fdd`
- frozen runner SHA-256: `923541554cd54c0b31ba1fca462946a08be53a1a9e68bcb91d33bb26adae5462`
- independent auditor SHA-256: `9eae76a5210add12973498ae2932f9bc5c0854814f0543457f0d706adadbaaf8`
- independent audit-result SHA-256: `577032fa962b105bb100d384a1ea52fad02fbd814f3d9282e03dff7e8399b031`
- independent audit: 13/13 checks passed
- all 807,032 accepted signal rows reconstructed exactly
- provider prices, triggers, fills, costs, sleeve compounding, daily returns, quarters, stock deletions, bootstraps, and decision reconstructed
- exact execution and daily-return reconstruction error: `0.0`
- full workspace research suite: 289 tests passed

Artifact root: `/private/tmp/stocker_frozen_regime_loop_pnl_sanity_v1_20260712`

The artifact root is ephemeral and should be archived before reboot if exact replay without recomputation is required.
