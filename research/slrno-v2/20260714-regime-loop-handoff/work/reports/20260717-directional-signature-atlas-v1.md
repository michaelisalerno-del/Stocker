# Directional Signature Atlas V1

## 1. Exact hypothesis

Small, interpretable combinations of information known by a completed fixed decision bar may repeatedly precede an economically meaningful upward move, downward move, or neither. A provisional directional lead had to retain the same sign and positive net payoff through discovery, validation, and the final opened period, survive twice costs and one-bar delay, and remain broad across sessions, stocks, months, nearby bins, and contributor deletions.

## 2. Difference from Long/Short/Neutral Detector V1

The prior detector used the same fixed-clock idea but estimated one general multinomial logistic equation against a first-touch target. This experiment instead used a fixed-terminal economic target, separately censused one- to three-condition rules and short state motifs, applied bounded shallow-tree proposal, froze discovery rules before validation, controlled multiplicity, kept movement permission separate from direction, and evaluated each signature independently before constructing an abstaining controller.

Repository inspection found pieces of this design in earlier experiments, but not this exact experiment. Selective Payoff Equations V1 studied selected payoff equations; Regime Utility Ablation V1 tested regime information; Loop Burst Mechanism V1, Causal Loop State Paths V1, Dynamic Loop × Regime Profitability V1, Dynamic Loop Temporary Payoff Edge State V2, Sequential Loop Competitor Veto V1, and Directed Economic Loop–Regime Rotation V1 studied structural mechanisms or selected populations; Fixed One-Bar Entry Latency V1 studied execution timing. None combined an unfiltered fixed-clock population, fixed-terminal labels, bounded one- to three-condition signature census, chronology freeze, multiplicity control, relative Track B, and append-only prospective logging. The machine-readable proof is `prior_experiment_coverage.json`.

## 3. Scientific status

All 2024, 2025, and partial-2026 observations are already opened retrospective data. The experiment is research-only, has `execution_enabled=false`, and supports neither a tradable-edge claim nor prospective validation. The final opened period was used only for unchanged scoring.

## 4. Available data periods

Exact EODHD regular-session five-minute bars produced 24,410 opportunities and 24,096 complete targets:

| Period | Population | Scored | Unavailable | Sessions | Stocks | LONG | NEUTRAL | SHORT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2024 discovery | 9,931 | 9,780 | 151 | 252 | 20 | 4,315 | 954 | 4,511 |
| 2025 validation | 9,926 | 9,815 | 111 | 250 | 20 | 4,423 | 918 | 4,474 |
| 2026 opened final | 4,553 | 4,501 | 52 | 122 | 19 | 2,121 | 439 | 1,941 |

AAL was excluded from 2026 to preserve the prior fixed-clock boundary. VTI-dependent fields stop on 2026-06-26; affected 2026-06-29 opportunities remain present with those features unavailable. Exact sector membership, sector-relative returns, quotes, spread, and order-book fields were unavailable and were not reconstructed from substitutes.

## 5. Fixed decision population

The population is one opportunity per eligible stock at decision ordinals 12 and 36: 10:30 and 12:30 America/New_York. It uses the exact 09:30-inclusive, 16:00-exclusive five-minute grid and no loop, setup, regime, direction, or profitability filter. Missing exact paths remain `UNAVAILABLE`.

## 6. Entry, terminal, and cost conventions

Entry is the exact provider open at decision ordinal plus one. Exit is the provider close at decision ordinal plus 24, without a restarted horizon. Entry cost is 5 bps and exit cost is 5 bps, for 10 bps round trip. Market impact and short borrow are not modelled.

## 7. Long/short/neutral target construction

Let `g` be the gross next-open-to-terminal long return in bps. Net long payoff is `g - 10`; gross short is `-g`; net short is `-g - 10`.

- `LONG`: `g > 20` and net long payoff is positive.
- `SHORT`: `g < -20` and net short payoff is positive.
- `NEUTRAL`: neither side qualifies.
- `UNAVAILABLE`: exact entry, terminal, complete path, or data quality is missing.

Future high-low range and long/short MFE and MAE are outcome-only diagnostics and never enter the feature ledger. The separate first-touch ledger uses the frozen ±20 bps barriers and preserves same-bar dual touches.

## 8. Neutral dead band

The primary dead band is two times the exact 10 bps round-trip cost: ±20 bps gross. One-times-cost (±10 bps) and three-times-cost (±30 bps) are sensitivity targets only. Internal construction is symmetric, and automated tests prove that a row cannot be both LONG and SHORT.

## 9. Movement-permission definition

The direction-neutral gate is the frozen `frozen_loop_movement_shadow_v1` h24 future-range point prediction at an exact causal state-run entry. There is no lower confidence bound, so the predeclared rule is `predicted_future_range_bps_h24 > 30`, three times round-trip cost. It passes on 5,653 opportunities and fails closed or is unavailable on 18,757. The gate is sparse because the retained prediction exists only when a state run begins on the completed fixed decision bar.

Movement permission did not improve directional economics: movement-permitted momentum averaged -9.714 bps per output in 2025 versus -7.844 for ordinary momentum, and -13.058 versus -7.448 in 2026.

## 10. Feature families

The sealed ledger contains 44 enabled causal features spanning structural state/history, compact loop summaries, price location and displacement, bar shape, range/volatility, movement permission and cost context, contemporaneous cross-section, breadth/dispersion, and broad clock phase. Every feature has an availability timestamp no later than its decision timestamp. Full raw loop-score vectors, stock/month identity, hindsight episodes, future routes, child/morph identity, targets, MFE, and MAE are prohibited.

Ambiguities fail closed. Loop summaries, departure probability, and movement predictions appear only at their exact causal state-run anchors; sector-relative and quote-derived features are explicitly unavailable; historical activity is only a causal proxy for liquidity.

## 11. State-motif construction

Motifs of lengths two, three, and four are reconstructed from causally observed state runs completed or known by the decision bar. A motif is one condition. No future state can enter a current motif. The independent audit reconstructed 22,172 motif rows.

## 12. Candidate-signature limits

Rules contain at most three conditions. The frozen caps were 2,000 total univariate/pairwise proposals before support filtering, 500 three-condition proposals, 100 shallow-tree proposals at depth at most three and leaf size at least 80, 100 retained discovery-stage candidates, 10 frozen discovery rules per direction, and 5 validation survivors per direction. Caps could not increase after scoring.

## 13. Support requirements

A rule required at least 80 rows, 30 sessions, 8 stocks, 3 calendar months, and 15 outcomes in its tested direction. No stock could provide more than 25% of rows. The same support rules applied in validation; failures remain in the candidate registry with explicit reasons.

## 14. Discovery score

The frozen score was:

`((mean_net_bps / 10) + 2*directional_lift) * chronological_consistency * breadth_factor * support_factor * cost_survival * neighbourhood_stability - concentration_penalties - 0.15*condition_count - opposite_direction_excess`

Chronological consistency is the fraction of positive discovery months; breadth saturates at 12 stocks and 60 sessions; support saturates at 200 rows; twice-cost survival receives full weight, primary-only survival half weight; immediately adjacent ordered bins determine neighbourhood stability. Ties prefer support, then fewer conditions, then signature ID.

## 15. Chronological split

Discovery is calendar 2024, validation is calendar 2025, and final opened scoring is 2026-01-01 through 2026-06-29. Discovery alone generated rules and any learned normalisation. Validation outcomes could only reject unchanged rules. No validation or final outcome regenerated rules or changed thresholds.

## 16. Number of candidates examined

Track A examined 2,503 candidates: 390 univariate, 1,610 pairwise, 500 three-condition, and 3 shallow-tree rules. Of these, 125 had support and a positive discovery effect, but zero passed broad FDR. The frozen 10-long/10-short library is therefore explicitly exploratory and exists to test chronological portability, not as a discovery-qualified family.

Track B separately examined 2,506 candidates.

## 17. Multiplicity controls

Broad discovery used Benjamini–Hochberg at q=0.10 on the more conservative of one-sided session-level payoff and lift tests. The retained family used Holm at alpha=0.10 and 2,000-draw, five-session-block bootstrap intervals. All failed candidates and rejection reasons are exported. Every Track A frozen directional rule had q=0.999955; none passed Holm in validation.

## 18. Strongest discovery long signatures

These were the five highest-ranked frozen exploratory long rules:

| Conditions | Rows | 2024 mean net bps | Long lift | 2025 mean net bps | 2025 twice-cost bps |
| --- | ---: | ---: | ---: | ---: | ---: |
| state 7 + clock 12 | 278 | +107.798 | +0.0804 | -25.924 | -35.924 |
| state 7 + session return above +1 causal scale | 193 | +94.690 | +0.0925 | -4.585 | -14.585 |
| state 7 | 456 | +77.233 | +0.0698 | -9.719 | -19.719 |
| normal compression + low departure probability | 184 | +88.184 | +0.0805 | -23.196 | -33.196 |
| state 7 + six-bar return above +1 causal scale | 97 | +94.048 | +0.1052 | -41.016 | -51.016 |

Discovery effects were large but multiplicity-unqualified and did not port.

## 19. Strongest discovery short signatures

| Conditions | Rows | 2024 mean net bps | Short lift | 2025 mean net bps | 2025 twice-cost bps |
| --- | ---: | ---: | ---: | ---: | ---: |
| state age 7+ + motif `5>6` | 123 | +52.777 | +0.0997 | +3.478 | -6.522 |
| movement permission false + motif `6>5>6` | 124 | +48.411 | +0.0952 | -39.601 | -49.601 |
| middle historical activity + motif `6>5>6` | 93 | +45.739 | +0.1194 | -59.098 | -69.098 |
| state 5 + bidirectional top-loop orientation | — | +29.540 | +0.0268 | -11.107 | -21.107 |
| bearish body + stock/market below -1 scale + strong breadth | — | +28.096 | +0.0745 | -27.143 | -37.143 |

The leading short rule retained positive primary-cost validation payoff, but failed twice costs, delay, contributor deletion, and Holm correction.

## 20. Validation results

No long and no short signature survived 2025. Among the 10 long rules, 3 had positive primary payoff, 9 retained positive lift, and 3 survived twice costs, but none met all requirements. Among the 10 short rules, the corresponding counts were 4, 9, and 2. All Holm-adjusted p-values were 1.0. This is a chronological rejection, not a lack-of-candidate fallback.

## 21. Final opened-holdout results

Because validation produced no survivor, no directional signature was opened for formal 2026 scoring and both final signature-metric files are intentionally empty. Diagnostic stress rows for the unchanged exploratory rules were retained without re-ranking. For example, state 7 + clock 12 rebounded to +14.364 bps in 2026 but was negative in validation and turned -6.456 bps under one-bar delay; the leading short rule was -34.886 bps in 2026. Neither can be interpreted as persistent.

## 22. Long signature library

The final long library is empty. Ten exploratory discovery rules remain frozen in the discovery library for auditability only.

## 23. Short signature library

The final short library is empty. Ten exploratory discovery rules remain frozen in the discovery library for auditability only. Short rules were not generated as inverses of long rules.

## 24. Neutral-veto results

Four neutral-veto conditions survived 2025 with positive lift and Holm significance:

| Condition | 2024 neutral lift | 2025 neutral lift | 2026 neutral lift |
| --- | ---: | ---: | ---: |
| previous state 1 + stock/universe return below -1 scale | +0.0735 | +0.0635 | +0.0634 |
| high cross-sectional dispersion + middle realised volatility | +0.0309 | +0.0410 | +0.0317 |
| middle realised volatility | +0.0316 | +0.0448 | +0.0268 |
| previous state 1 | +0.0455 | +0.0259 | -0.0006 |

Three of four retained the lift sign in 2026; `realised_volatility == middle` was the only one satisfying the final implementation's full neutral-veto checks. This evidence identifies comparatively directionless conditions more reliably than a positive directional action. It remains opened retrospective evidence.

## 25. Atlas-level controller result

The controller uses one rule/one vote, any opposing vote forces neutral, missing movement permission forces neutral where required, and non-positive conservative value forces neutral. With zero validated directional signatures it issued 0 LONG, 0 SHORT, and 100% NEUTRAL in both 2025 and 2026. Total payoff and cost were zero. This equals always-neutral and is the correct fail-closed result; it is not evidence of directional skill.

Predictive metrics reflect base-rate probabilities with abstention: 2025 macro Brier was 0.247865 and log loss 0.939152; directional precision and recall were undefined/zero. AUC was 0.5 and was not used to claim success.

## 26. Comparison with momentum and reversal

| Model | 2025 net bps/full opportunity | 2026 net bps/full opportunity |
| --- | ---: | ---: |
| Atlas | 0.000 | 0.000 |
| One-bar momentum | -7.674 | -7.327 |
| One-bar reversal | -11.892 | -12.349 |

The atlas loses no money only because it abstains everywhere. It did not produce directional outputs that economically outperform these baselines.

## 27. Comparison with static price-context model

The prior static price-context multinomial baseline lost -12.143 bps per opportunity in 2025 and -17.050 in 2026. Its 2025 macro Brier was 0.248494 and log loss 0.940703, both slightly worse than the abstaining atlas probabilities, but this is not actionable discrimination.

## 28. Comparison with state and state history

Current state alone lost -10.419 bps per opportunity in 2025 and -11.877 in 2026. Current state plus history lost -13.654 and -8.368. State motifs did generate attractive discovery candidates, but none passed the unchanged validation gate. The atlas therefore found no portable directional increment over state/history.

## 29. Movement-permission increment

Movement permission passed 2,291/2,253/1,109 rows in 2024/2025/2026. It did not clarify direction: its momentum combination remained negative and the strongest exploratory signatures did not become stable. The retained range model may describe a narrow state-run surface, but it does not supply a portable directional gate at these clocks.

## 30. Cross-sectional Track B

Track B completed after Track A using equal-universe residual returns and fixed top-20%/middle-60%/bottom-20% labels. Sector-relative scoring was disabled because exact frozen sector membership was unavailable. One of 2,506 discovery candidates passed FDR: clock 12 + high historical activity, with 915 rows, +34.112 bps mean residual and +0.0848 relative-long lift. Applied unchanged in 2025 it had 1,290 rows, -5.140 bps mean residual, failed sign, payoff, twice-cost, month consistency, and Holm, and did not survive.

The contemporaneous relative-strength baseline averaged +1.816 residual bps per output in 2024, +2.136 in 2025, and -3.089 in 2026. No portfolio/cost translation was made, so no absolute-profitability claim is permitted. Relative direction was not more persistently predictable than absolute direction.

## 31. Twice-cost result

No directional signature survived twice-cost validation as part of the full survival rule. The leading short rule moved from +3.478 bps at primary cost to -6.522; the leading long rule was already -25.924 and became -35.924. Atlas outputs remained zero.

## 32. One-bar-delay result

Across frozen rules, 5/10 long and 3/10 short rules were positive under 2025 delay, with median delayed means of +0.309 and -12.423 bps respectively, but none met the complete validation criteria. The leading short rule fell to -10.825 bps. The 2026 rebound of the leading long rule fell to -6.456 bps.

## 33. Stock and month stability

After removing the top five stocks, 0/10 long and 0/10 short validation rules had positive mean payoff. After removing the top five hindsight episodes, 0/10 long and 2/10 short rules remained positive, but neither short passed the other criteria. The leading short rule moved from +3.478 to -5.021 bps after its best stock, and to -40.585 after its top five stocks. These are material instability failures.

## 34. Concentration

Concentration was exported by signature, stock, period/month, clock, state, motif, loop family, sector availability, and hindsight episode. The leading discovery long rule had top-stock/top-month absolute contribution shares of 0.211/0.377; the leading short had 0.143/0.215. These point shares were not individually disqualifying in discovery, but validation sign reversals and top-contributor deletion failures prevent retention. HHI and top-five shares are in `concentration_results.csv`.

## 35. Failed or unstable signatures

The complete candidate registry preserves all 2,503 Track A rules and explicit rejection reasons. The strongest apparent rules were state 7 long variants and state-history short motifs. Their discovery effects were multiplicity-unqualified, changed sign or failed costs in validation, and were sensitive to clock, delay, or contributor deletion. Seven null families each produced zero persistent-positive rules, so no null created a replacement signature.

## 36. Did any telltale sign persist?

No simple long or short telltale sign met the predeclared persistence criteria. Directionless conditions—especially middle realised volatility—were more stable than positive direction. The causal information gathered by Stocker may contain descriptive directional associations, but this experiment found no small portable rule that survived chronology and economic stress.

## 37. Was relative direction more predictable?

No. Track B's sole multiplicity-qualified discovery signature failed unchanged validation, its final library is empty, and the simple relative-strength baseline reversed from small positive residuals in 2024–2025 to negative in 2026.

## 38. Prospective logging implementation

An execution-free, hash-chained `ProspectiveLedger` writes separate append-only forecast and settlement JSONL streams. Forecast construction consumes an outcome-free feature row, records all availability timestamps and frozen signature decisions, and applies movement/conflict/conservative-value logic. Settlement is a distinct immutable record and cannot overwrite a forecast. Duplicate opportunity IDs and opened historical rows fail closed. Dry runs wrote one forecast and one settlement and passed independent chain verification.

The frozen completion rule requires 2,000 settled opportunities, 100 sessions, 15 stocks, four months, 100 LONG and 100 SHORT outputs, and 30 sessions containing each direction. Because both directional libraries are empty, the present atlas cannot meet its directional output counts and should not be promoted into a directional prospective campaign. The logger can remain available for a future separately frozen library or blinded neutral-only administrative logging.

## 39. Scientific decision

`neutral_veto_more_reliable_than_direction`

This label reflects the replicated neutral-lift evidence and the absence of any validated long or short library. It does not approve trading, paper trading, deployment, or prospective edge claims.

## 40. Exact next recommendation

Freeze and prospectively log the causal feature ledger and neutral-veto states without positions or P&L-driven adaptation, while separately improving the availability and calibration of the direction-neutral movement/range layer at fixed clocks. Do not deepen the rule search. Only revisit directional signatures after a new, untouched completion window exists and exact movement predictions cover the fixed-clock population broadly enough to test movement permission without sparse-anchor confounding.

## Reproducibility and identity

- Run ID: `directional-signature-atlas-v1-8916f0cceb4c-09f6cae0d89b`
- Git SHA: `515d5cf002e33c3cdc000d47cc433205ffcde060`
- Contract SHA-256: `8916f0cceb4ce0c791973c6474f6f263c053e09bd5d072b79ecdf3a99c7776b9`
- Feature-schema SHA-256: `905e8634eeeb362d4abbd1fd04b90526c7c0561c7c5e9cc7d87779088f1d5a7b`
- Data-snapshot SHA-256: `09f6cae0d89bb99973b1df410df579e4c9b6e33e0bd90b4f3f65dab1627e30aa`
- Sealed feature-ledger SHA-256: `001f34ff630c5426607bc39e9247735c9d4a7374f29f5d337634b6893983f0ff`
- Independent audit: 22/22 passed
- Exact rerun: 67 artifacts byte-identical; no missing, extra, or mismatched files
- Safety: research-only; broker, IG, orders, positions, exits, deployment, runtime, keys, and secrets unchanged
