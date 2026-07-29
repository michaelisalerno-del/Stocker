# Daily Stock + Front-Options Context Quick Screen V0.1

## Result

Overall decision: `stock_structure_improves_iv_excess_only`.

Branches were run independently. Branch A used the full frozen structural panel;
Branch B used exact previous-close front options without term structure; Branch C
tested underlying 15-minute movement relative to prior-close ATM IV. No option P&L
was calculated.

Component statuses:

- Daily stock context: `not_supported`
- Front-options regimes: `descriptive_only`
- Front-options completion context: `not_supported`
- Stock structure to IV excess: `supported`
- Broad-conflict IV residual: `descriptive_only`
- Back-expiry schema preflight: `supported_noncompact_schema`

Structural reconstruction passed: `True`; clean rows:
`87443`; assessment rows:
`34577`; row, route-state, and target mismatches:
`0`,
`0`, `0`.

## Branch A

Support: `{"maximum_weighted_stock_share": 0.05180712176493042, "months": 8, "passed": true, "positive_outcomes": 433, "rows": 33729, "sessions": 159, "stocks": 20}`.

| model | log_loss | brier_score | auc | average_precision | base_rate | rows | sessions | stocks | positive_outcomes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0 | 0.10672868 | 0.01922221 | 0.71473278 | 0.04249964 | 0.01803732 | 33729 | 159 | 20 | 433 |
| A1 | 0.10666266 | 0.01922105 | 0.71391535 | 0.04157936 | 0.01803732 | 33729 | 159 | 20 | 433 |

## Branch B

Support: `{"broad_conflict_rows": 2132, "low_route_support_rows": 2883, "maximum_weighted_stock_share": 0.0775916001699797, "months": 8, "passed": true, "positive_outcomes": 142, "rows": 10265, "sessions": 154, "stocks": 20}`.

| model | log_loss | brier_score | auc | average_precision | base_rate | rows | sessions | stocks | positive_outcomes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0 | 0.16083245 | 0.02824916 | 0.67101022 | 0.03495832 | 0.01858853 | 10265 | 154 | 20 | 142 |
| B1 | 0.15973654 | 0.02801813 | 0.66333872 | 0.03369107 | 0.01858853 | 10265 | 154 | 20 | 142 |

## Branch C

Support: `{"broad_conflict_rows": 2132, "low_route_support_rows": 2883, "maximum_weighted_stock_share": 0.0775916001699797, "months": 8, "passed": true, "positive_outcomes": 2921, "rows": 10265, "sessions": 154, "stocks": 20}`.

| model | log_loss | brier_score | auc | average_precision | base_rate | rows | sessions | stocks | positive_outcomes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | 0.58655429 | 0.19921836 | 0.61141718 | 0.37755012 | 0.28546003 | 10265 | 154 | 20 | 2921 |
| C1 | 0.57207160 | 0.19314455 | 0.65462995 | 0.42687388 | 0.28546003 | 10265 | 154 | 20 | 2921 |

### Primary increments

| branch | comparison | log_loss_improvement | brier_improvement | auc_improvement | average_precision_improvement |
| --- | --- | --- | --- | --- | --- |
| A | A1-A0 | 0.000066022 | 0.000001154 | -0.000817428 | -0.000920281 |
| B | B1-B0 | 0.001095911 | 0.000231027 | -0.007671501 | -0.001267254 |
| C | C1-C0 | 0.014482685 | 0.006073806 | 0.043212775 | 0.049323769 |

### IV-relative movement by frozen route state

| route_state | rows | mean_absolute_movement | median_absolute_movement | mean_iv_expectation | mean_iv_residual | median_iv_residual | exceed_iv_rate | iv_sigma_ratio | upper_decile_iv_residual | top_5pct_positive_residual_contribution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BROAD_CONFLICT | 2132 | 0.00456071 | 0.00351306 | 0.00754740 | -0.00298669 | -0.00342847 | 0.19424794 | 0.48214226 | 0.00216080 | 0.59833344 |
| LOW_ROUTE_SUPPORT | 2883 | 0.00574509 | 0.00422678 | 0.00844141 | -0.00269633 | -0.00357724 | 0.22036328 | 0.54302703 | 0.00365005 | 0.59144990 |
| NARROWING | 251 | 0.00942492 | 0.00703861 | 0.00946417 | -0.00003925 | -0.00203494 | 0.41204297 | 0.79457543 | 0.00884383 | 0.50930472 |
| OTHER | 4999 | 0.00833862 | 0.00642123 | 0.00938124 | -0.00104263 | -0.00258576 | 0.35291629 | 0.70920804 | 0.00827023 | 0.44452129 |

The binding BROAD_CONFLICT-minus-LOW_ROUTE_SUPPORT mean residual is
`-0.000290361`.

## Daily-stock reconstruction and regimes

Daily-stock reconstruction passed: `True`. Assessment
feature retention: `97.647807%`.
Maximum dimension difference:
`0.0`; maximum
posterior difference:
`4.884981308350689e-15`.

| regime | hard_rows | months | posterior_mass | sessions | stocks | supported |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 445 | 8 | 0.147035 | 123 | 20 | True |
| 1 | 835 | 8 | 0.274347 | 155 | 20 | True |
| 2 | 370 | 8 | 0.122773 | 102 | 20 | True |
| 3 | 1422 | 8 | 0.455845 | 158 | 20 | True |

| regime | daily_activity_acceleration | daily_compression | daily_directional_efficiency | daily_extension | daily_rejection | daily_relative_strength | daily_trend_persistence | daily_volatility_acceleration |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.9946 | -0.5686 | 0.5769 | 1.2884 | -0.4206 | 1.3761 | 0.9860 | 0.2830 |
| 1 | 0.4851 | -0.4240 | -0.3038 | 0.0834 | 0.1792 | 0.0058 | -0.0103 | 0.6130 |
| 2 | 0.2748 | 0.0082 | 0.6521 | -0.8960 | -0.4858 | -0.8024 | 0.8021 | -0.1210 |
| 3 | -0.3695 | 0.3235 | -0.1277 | -0.0329 | 0.1574 | -0.0978 | 0.0341 | -0.3750 |

## Front-options regime centroids

| regime | front_options_implied_tension | front_options_premium_richness | front_options_downside_asymmetry | front_options_liquidity_stress | front_options_positioning_concentration |
| --- | --- | --- | --- | --- | --- |
| 0 | -0.2716 | -0.2750 | 0.0529 | -0.1726 | 0.2007 |
| 1 | 0.2658 | 0.2758 | -0.9123 | 1.3425 | 0.4012 |
| 2 | 0.4330 | 0.3985 | -0.6354 | 0.9108 | -0.0912 |
| 3 | 0.7492 | 0.6957 | 40.5237 | 2.1019 | -1.1910 |

| regime | hard_rows | months | posterior_mass | stocks | supported |
| --- | --- | --- | --- | --- | --- |
| 0 | 725 | 8 | 0.777142 | 20 | True |
| 1 | 51 | 8 | 0.054898 | 9 | True |
| 2 | 153 | 8 | 0.167960 | 18 | True |
| 3 | 0 | 0 | 0.000000 | 0 | False |

Front pairs reconstructed: `2179` stock-sessions (`1250` development, `929` assessment). Same-day or future observations: `0`. Selection was rebuilt from the repaired cached chains: `True`; selected-contract mismatches: `0`. All eight front raw features had finite development support: `True`.

## Mismatch distributions

| feature | maximum | mean | median | minimum | q10 | q90 | rows | standard_deviation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mismatch_complacent_broad_conflict | 2.419958 | 0.105444 | 0.000000 | -2.673716 | 0.000000 | 0.614261 | 10265 | 0.384356 |
| mismatch_compression_vs_front_iv | 4.118519 | 0.258487 | 0.325719 | -4.035483 | -1.642395 | 2.003437 | 10265 | 1.397305 |
| mismatch_daily_volatility_vs_front_iv | 4.173719 | 0.281597 | 0.239895 | -3.734662 | -1.061142 | 1.693031 | 10265 | 1.139720 |
| mismatch_direction_agreement | 5.727371 | -0.043680 | -0.009872 | -7.175409 | -0.705507 | 0.537030 | 10265 | 0.701376 |
| mismatch_route_vs_front_premium | 4.667233 | 0.228029 | 0.224214 | -4.721068 | -1.489338 | 2.135509 | 10265 | 1.397061 |

## Monthly, checkpoint, bootstrap, and null stability

| branch | positive_log_loss_months | months | minimum_monthly_log_loss_improvement | maximum_monthly_log_loss_improvement |
| --- | --- | --- | --- | --- |
| A | 5 | 8 | -0.000263955 | 0.000584573 |
| B | 8 | 8 | 0.000012347 | 0.001486958 |
| C | 8 | 8 | 0.003485902 | 0.022203406 |

| branch | checkpoint_group | log_loss_improvement | brier_improvement |
| --- | --- | --- | --- |
| A | early_6_14 | 0.000055317 | -0.000000472 |
| A | middle_16_24 | 0.000054669 | -0.000001327 |
| A | late_26_34 | 0.000095369 | 0.000006418 |
| B | early_6_14 | 0.000823488 | 0.000172069 |
| B | middle_16_24 | 0.001045749 | 0.000217475 |
| B | late_26_34 | 0.001562591 | 0.000334895 |

The complete 80% interval surface is:

| statistic | lower | upper |
| --- | --- | --- |
| A1_minus_A0_log_loss_improvement | -0.000019581 | 0.000138259 |
| A1_minus_A0_brier_improvement | -0.000008963 | 0.000008376 |
| A1_minus_A0_auc_improvement | -0.003530170 | 0.001822797 |
| A1_minus_A0_average_precision_improvement | -0.002524479 | -0.000240665 |
| B1_minus_B0_log_loss_improvement | 0.000975785 | 0.001339143 |
| B1_minus_B0_brier_improvement | 0.000203959 | 0.000280617 |
| B1_minus_B0_auc_improvement | -0.011638735 | -0.005079465 |
| B1_minus_B0_average_precision_improvement | -0.002817022 | -0.000509842 |
| C1_minus_C0_log_loss_improvement | 0.012452877 | 0.015901544 |
| C1_minus_C0_brier_improvement | 0.005178479 | 0.006718983 |
| C1_minus_C0_auc_improvement | 0.033670676 | 0.047149091 |
| C1_minus_C0_average_precision_improvement | 0.039925173 | 0.059476374 |
| BROAD_CONFLICT_minus_LOW_ROUTE_SUPPORT_mean_iv_residual | -0.000535684 | -0.000020331 |
| BROAD_CONFLICT_minus_LOW_ROUTE_SUPPORT_median_iv_residual | -0.000035226 | 0.000389650 |
| BROAD_CONFLICT_minus_LOW_ROUTE_SUPPORT_exceed_iv_rate | -0.040109595 | -0.010571556 |

| null | refits | log_loss_improvement | brier_improvement | auc_improvement | average_precision_improvement |
| --- | --- | --- | --- | --- | --- |
| front_options_bundle | 3 | 3 | 3 | 0 | 1 |
| stock_structure_bundle | 3 | 3 | 3 | 3 | 3 |

## Resampling and nulls

The bootstrap contains `15` statistics and exactly
ten fixed-seed, fixed-prediction, whole-session draws with 80%, 90%, and 95%
intervals. Front-options null refits: `3`. Stock-structure null
refits: `3`. No bootstrap draw refit a model.

Maximum concentration:

| population | concentration_type | group | weighted_rows | share |
| --- | --- | --- | --- | --- |
| branch_a | month | 2025-07 | 21.592105 | 0.138176 |
| branch_a | stock | SMCI | 8.095666 | 0.051807 |
| joined_front_options | month | 2025-07 | 7.729825 | 0.165779 |
| joined_front_options | stock | RIOT | 3.617888 | 0.077592 |

## Back-expiry schema preflight

Status: `supported_noncompact_schema`. Endpoint: `/mp/unicornbay/options/eod`.
Exactly `1` request returned
`16` records;
`9` were exact-date rows and
`9` were 46–90 DTE rows. Exact-date
filtering was possible: `True`.
Canonical cache modified: `False`.
The provider returned
`7` protected-date records despite the exact
request; they were rejected without persistence, and protected records persisted
were `0`. The future plan preserves
non-compact identities and exact-date filtering; it is not a DTE recommendation.

Model branches reused
`139622`
cached provider records and downloaded zero model-branch records.

## Chronology and protection

All admitted option observations are from the exact prior US trading session. No
same-day or future option record was used. Protected market and option observations
dated 2025-08-23 or later are zero.
Materialised protected market rows:
`0`; protected option observations:
`0`.

## Reproducibility

Determinism passed: `True`. Joined-row mismatches:
`0`. Maximum feature
difference: `0.0`.
Maximum probability difference:
`0.0`. Options were
not redownloaded, and bootstrap/null draws were not repeated.

Independent audit passed: `True`
(`21/21` checks). Audit status:
`passed`. The auditor made zero
provider requests and refit zero null models.

## Plots

- `/Users/michaelsalerno/Documents/Codex/2026-07-23-you-are-working-in-the-github-3/research/cross-market-context/20260723-daily-stock-front-options-context-v01/reports/completion_model_proper_scores.png`
- `/Users/michaelsalerno/Documents/Codex/2026-07-23-you-are-working-in-the-github-3/research/cross-market-context/20260723-daily-stock-front-options-context-v01/reports/iv_excess_and_route_state_residuals.png`

## Scientific boundary

This retrospective screen does not establish option profitability, intraday option
fills, economic or directional edge, prospective validity, trading utility, or a
deployable strategy.
