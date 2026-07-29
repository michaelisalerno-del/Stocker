# Profitable Loop Episode Anatomy V1

**Run ID:** `profitable-loop-episode-anatomy-v1-frozen-run`

**Contract:** `20260717-profitable-loop-episode-anatomy-v1`

**Frozen source Git identity:** `d199bed1e1d66199ba63b3f5e12df03768728484` on `agent/slrno-research-handoff`

**Primary horizon:** frozen 24 bars

**Status:** read-only, retrospective mechanism research; no predictor, selector, threshold, admission rule, or trading rule was built

**Primary scientific decision:** `coactivation_not_above_null`

The complete machine-readable result is under `work/artifacts/20260717-profitable-loop-episode-anatomy-v1/primary/`; the byte-identical rerun is under `exact_rerun/`. All bps figures below are descriptive net payoff unless labelled otherwise.

## 1. Registered mechanism hypotheses

The registered primary hypothesis was that temporary profitable loop periods are frequently shared economic episodes affecting several loop–regime pairs, while one loop orientation captures a moderately dominant payoff share. Its dominance could come from occurrence frequency, payoff per occurrence, or both. The second hypothesis was that completed prior regime paths may change the payoff meaning of an otherwise unchanged loop/current-regime pair. The third was that state/history may explain broad common movement while orientation explains leader-specific excess.

These are anatomy hypotheses. Hindsight-positive labels define historical windows only and are never causal inputs.

## 2. Difference from selector and prediction experiments

Earlier dynamic-state, lead/lag, veto, rotation, detector, atlas, branch, admission, child/morph, OR-stack, and payoff-equation work asked whether an observable state could select, predict, or veto future economics. This experiment instead synchronises the already-opened historical outcomes, reconstructs their episode boundaries, and decomposes what happened. It fits no predictive model and produces no deployable decision surface.

## 3. Scientific status

The 2023, 2024, 2025, and partial-2026 research surfaces have already been opened. They are not prospective validation. The retained immutable positive-episode population used by the census contains 2023 and 2025 only; those two periods are historical replications of anatomy, not holdouts. Nothing here authorises live, paper, or demo trading, orders, positions, sizing, execution changes, exit changes, or strategy promotion.

## 4. Source data, identity, and missing inputs

The contract hashes 13 retained immutable Parquet sources covering the V2 payoff/episode/causal-feature/trade panels, rotation same-regime attribution, causal route events, sequential episode/path diagnostics, and T0 named/control/envelope references. The data-snapshot identity is `bebb812b038d17ba4bf3f48a17a9205276657cdf04d7f1df630c65d1f80e8a8b`; the contract hash is `87c6b0d6349dfb6607f26ec369308cb853bc2ce6d83c72e30fd6928c39ef707b`.

The original 2023/2024/2025 anchor-provider paths, accepted-signal ledger, execution manifest, 2023 provider root, and formal Frozen Named-Loop T0 Markdown report were not retained. On the matched primary population, regime-history length four, causal state age, completed dwell, same-orientation repeat count, VWAP/typical-price distance, opening-range position, sector, beta, and the raw one-to-three-bar path score were unavailable. They remain unavailable. Route-event next-open payoff and the subsequent increment to frozen close were available and are reported outcome-only. Sector was not inferred from company names and Atlas fields from a different population were not joined. A further 959 valid source payoff cells precede the retained trade-decision warm-up, so their payoff remains usable while occurrence/history fields remain missing. The occurrence tape contains 9,926 raw fills but exactly 9,548 frozen stock/session/loop/orientation occurrences. Of those, 349 contain conflicting full history tokens across repeated same-stock fills; each remains one occurrence. History availability fails closed separately by registered length: 58 of those 349 retain a unanimous length-two prefix even though their longer history conflicts, while a conflict at the requested length remains unavailable. At the synchronized pair-cell level, 953 rows contain a disagreement at one or both retained history lengths. Every contributing stock must agree on one non-missing token at that length: 988 panel rows retain length-two history and 736 retain length-three history.

## 5. Exploratory census reproduction

The frozen definition is exactly `hindsight_payoff_state == "positive"`; `decaying`, negative, and missing states are not positive. No threshold was changed.

| Census quantity | Reproduced |
|---|---:|
| Strict-positive pair rows | 674 |
| Positive sessions | 322 |
| Sessions with at least two positive pairs | 210 (65.2174%) |
| 2023 | 93/159 (58.4906%) |
| 2025 | 117/163 (71.7791%) |
| Same-regime episodes | 107 |
| Single-loop same-regime episodes | 65 (60.7477%) |
| Multi-loop same-regime episodes | 42 (39.2523%) |
| Multi-loop share, 2023 / 2025 | 38.0952% / 40.9091% |
| Multi-loop leader-share median | 64.9895% (41 available; 1 unavailable) |
| Majority leader / over-80% leader | 33 / 11 |

All requested figures reproduced exactly, so the extension gate opened.

## 6. Null-adjusted co-activation

The pair-specific Poisson-binomial independence null expected a conditional multi-pair share of about 64.10%. The primary 2,000-draw, seed-`20260717`, five-session block-circular null preserved period boundaries, eligibility masks, pair-positive counts, and within-pair persistence approximately. It expected 64.0818% (95% simulation interval 59.5808%–68.5190%). Observed was 65.2174%, an excess of only **1.1356 percentage points**, with one-sided empirical `p = 0.3153`.

Observed positive-session counts were 112 with exactly one pair, 114 with two, 59 with three, and 37 with four or more; the maximum was six. The raw 65.2% is therefore explained well by eligibility, pair rates, and persistence. The registered shared-episode claim does **not** clear its null requirement.

## 7. Frozen episode definitions

- Level 1 pair episode: the immutable contiguous positive episode for one loop × current regime/orientation pair.
- Level 2 same-regime episode: the union of overlapping or adjacent pair episodes in the same current regime.
- Level 3 shared-market episode: the corresponding union across regimes.

Adjacency means the same or consecutive retained trading sessions within a period. Missing sessions and period boundaries are not bridged; 2023 and 2025 never connect. Pre/on/post timelines use up to ten retained sessions on either side and record boundary truncation. A no-opportunity cell is missing, not zero or negative.

## 8. Episode counts and durations

There were 215 pair episodes and 107 same-regime episodes. In 2023, 63 same-regime episodes had median duration 8 sessions (mean 13.52); in 2025, 44 had median 9 (mean 24.80). The cross-regime adjacency union mechanically produced two period-wide Level-3 chains, each 190 sessions. Their extreme span is evidence that adjacency union alone is too permissive for claiming an economic common episode, especially because co-activation did not exceed the null.

## 9. Single-loop versus multi-loop anatomy

The 107 same-regime episodes classified deterministically as 65 `SINGLE_LOOP_EPISODE`, 11 `EXTREME_LOOP_DOMINANCE`, 22 `MAJORITY_LOOP_DOMINANCE`, 8 `DIFFUSE_MULTI_LOOP`, and 1 `UNKNOWN_MULTI_LOOP` because positive-share support was unavailable. Thus 60.7% were single-loop and 39.3% multi-loop. The multi-loop fraction was similar in 2023 and 2025, but this stability does not overcome the null result.

## 10. Leader-share distribution

Among 41 evaluable multi-loop episodes, median final-leader positive-payoff share was 64.99%; 33/42 multi-loop episodes had a majority leader and 11/42 exceeded 80%. Eight were genuinely diffuse and one was unknown. Ties are recorded explicitly and never broken lexically.

## 11. Occurrence dominance versus payoff efficiency

The multi-loop leader's median occurrence share was only 29.41%, versus median positive-payoff share of 64.99%. Total loop payoff was decomposed exactly as occurrence count × robust payoff per occurrence, against the pre-episode baseline where available; the independent reconstruction's maximum identity error was 2.27e-13 bps. All 107 same-regime episodes had complete stock-occurrence coverage for their supported episode payoff cells. Repeated fills from one stock were collapsed to one capped stock/session/loop/orientation occurrence even when their history tokens differed, and raw fill count never weighted the economic decomposition.

## 12. Leader efficiency

`leader_efficiency = positive-payoff share / occurrence share`. Median efficiency across evaluable multi-loop episodes was 2.284, and 70.73% exceeded one. Temporary dominance was therefore more often payoff-efficiency-heavy than pure frequency-heavy. This is descriptive attribution, not a payoff predictor. Zero-occurrence denominators remain unavailable.

## 13. Common episode component

For supported pair payoff `y(l,r,t)`, the primary common component is the equal-pair median `C(t)` across supported eligible pairs. Its global median was -0.289 bps. Every aggregate component share now expands C, R, and L over the same supported pair rows before summing. In multi-loop same-regime episodes, the median positive-component-mass share for common intensity was 25.53%; its median signed contribution to positive pair payoff was 16.04%, and its median non-causal marginal-variance share was 15.68%. The Level-3 2023/2025 chains had common mass shares of 24.14%/27.46%, not a dominant broad common-payoff explanation.

## 14. Regime component

`R(r,t)` is the median of `y(l,r,t) - C(t)` across supported loops in the same regime. In multi-loop same-regime episodes its median positive-component-mass share was 37.52%, its signed positive-payoff contribution was 34.10%, and its non-causal marginal-variance share was 35.97%. The two period-wide Level-3 chains had regime mass shares of 45.75% and 45.42%. These are nonlinear descriptive summaries of mechanically long unions, not causal variance attribution.

## 15. Loop-orientation excess

`L(l,r,t) = y(l,r,t) - C(t) - R(r,t)`, giving the exact row identity `y = C + R + L` to numerical precision. Median loop-excess positive-component-mass share in multi-loop episodes was 36.44%, its median signed positive-payoff contribution was 48.88%, and its non-causal marginal-variance share was 43.69%. Loop excess therefore carries the largest median signed payoff and marginal-variance attribution, while regime mass is marginally larger under positive-part normalisation. This mixed decomposition, together with leader efficiency, supports loop-specific realised excess in many episodes but gives no prospective identifier.

## 16. Component persistence

Session-lag-one rank correlations were weak: common 0.012/0.031, regime 0.041/0.056, and loop excess 0.017/-0.019 in 2023/2025. The equal-pair winsorised-mean sensitivity is retained separately and does not replace the median decomposition. No component shows persistence large enough to be called forecastability.

## 17. Same-regime anatomy

The frozen component rule identified 8 same-regime episodes as broad shared-regime activation and 33 as loop-specific activation. Regime and loop-excess attribution are both material, with the answer depending on whether positive mass, signed positive-payoff contribution, or non-causal marginal variance is summarised. These categories can overlap with concentration flags and are descriptive. They were frozen before final output and were not optimised to generate interesting cases.

## 18. Multi-regime shared-market anatomy

The two Level-3 unions spanned 10 loops/7 regimes in 2023 and 12 loops/8 regimes in 2025. Their final leader shares were only 25.10% and 20.45%, so they were diffuse; their common/regime/loop-excess positive-mass shares were 24.14%/45.75%/30.11% and 27.46%/45.42%/27.13%. Because the block null explains the session co-activation rate and adjacency merges nearly whole periods, these unions are timeline containers, not evidence of two genuine shared market episodes.

## 19. Early leader emergence

Across the 42 multi-loop episodes, checkpoint-evaluable prefixes numbered 32, 39, 41, 41, and 42. Tie-aware top-one matches to the frozen final leader were 50.00% after one session, 58.97% after two, 58.54% after three, 56.10% after the first quarter, and 59.52% after the first half. Top-three inclusion was 96.88%, 97.44%, 100.00%, 97.56%, and 97.62%; it is weakly discriminating in this small loop population. Each provisional ranking uses only its historical prefix; the frozen final ranking is used only for retrospective scoring. Episodes with no positive prefix do not receive invented lexical leaders.

## 20. Leader persistence

Across episode rows with supported, positive comparisons, tie-aware current-leader top-one persistence was 35.74%, 32.37%, and 31.82% at lags one, two, and three; top-three persistence was 63.84%, 60.94%, and 60.53%. Results separately encode same episode, boundary, no-positive-pair, and missing-support cases; zero-payoff sessions cannot create leaders. These rates describe realised continuity only.

## 21. Payoff remaining at checkpoints

For multi-loop episodes, median total episode payoff remaining was 94.47% after the first session, 91.23% after two, 81.41% after three, 75.50% after the first quarter, and 47.88% after the first half. Final-leader payoff remaining was 100.00%, 88.88%, 85.05%, 78.31%, and 51.27%. Fractions use realised supported positive payoff as their denominator and never exceed one. Although substantial payoff remained early, top-one identity was not stable enough, the calculation is hindsight-conditioned, and the co-activation prerequisite failed.

## 22. Regime-sequence census

Primary causal sequences use completed prior states of lengths two and three; length four is an unavailable sensitivity and was not substituted. Support required at least 30 stock-capped rows, 15 sessions, 8 stocks, 3 months, and no stock above 30% of rows. Thirty-four sequence groups cleared support. Every sequence and four-way group reports the frozen `change_session` / `after_change_1_3` / `other` clock-phase composition and availability; clock phase is descriptive context, not part of a searched sequence identity. Conflicting within-stock history tokens are unavailable rather than duplicated. Current state, loop identity, sequence timestamp, period, recurrence/clock availability, and all unsupported cells are retained in the census.

## 23. Four-way counterfactuals

For each target loop/current-state/sequence, the tables separate target-loop + target-sequence, target-loop + other-sequence, other-loop + target-sequence, and other-loop + other-sequence. These mutually intended groups report rows, sessions, stocks, occurrence, mean/median payoff, positive rate/payoff, C/R/L, exact twice-cost sensitivity, period stability, concentration, clock-phase composition, and support. Inference uses the registered 1,000-draw, five-session circular moving-block bootstrap over the within-period union session calendar, keeping all stock rows from a sampled session clustered. The empirical two-sided bootstrap p-values—not row-level independent tests—feed BH FDR. These are descriptive comparisons, not selected rules.

## 24. Sequence × loop interaction

Only one of the four comparison-supported groups had both directional increments positive: 2023 `cycle_11|state_5` after `state_4>state_5` (+53.83 bps sequence, block 95% interval -10.29 to +118.69; +8.74 bps within-sequence loop increment, -65.93 to +81.40). Its BH q-values within the four supported comparisons were 0.456 and 0.981. The other 410 descriptive comparisons retain counts and point estimates but their intervals, p-values, q-values, and FDR labels are unavailable. The analogous 2025 pair entered from a *different* `state_6>state_5` path and its within-sequence loop increment was negative. No comparison survived multiplicity control and no same frozen path had a controlled same-direction effect in both periods. The sequence-interaction hypothesis is not supported.

## 25. Named `cycle_04|state_4` anatomy

This named pair had 12 pair episodes embedded in 8 same-regime unions. On target-pair rows inside its own frozen pair episodes—never whole-episode peer averages—mean C/R/L were +10.88/+40.09/+43.03 bps. Mean occurrence share was 34.04%, positive-payoff share 45.74%, and median payoff efficiency 2.855. The retained T0 references were +30.75 bps at F0 and +20.73 at F10. Within its positive episodes, route composition was 18.70% exact completion, 65.04% incompatible first transition, and 16.26% expected-leg diversion. It covered eight reference months; top-stock contribution share was 25.78%. First-session final-leader match was 42.86% with median 97.45% of episode payoff remaining. Its four-test Holm-adjusted descriptive p-value was 0.595.

## 26. Named `cycle_07|state_5` anatomy

This named pair had 25 pair episodes embedded in 18 same-regime unions. Target-pair mean C/R/L were +13.09/+20.66/+30.00 bps. Its mean occurrence share was 77.33%, positive-payoff share 55.91%, and median efficiency 0.786: it was frequent but less efficient than its occurrence share implied. T0 F0/F10 were +17.04/+7.02 bps. Within positive episodes, route composition was 27.10% exact, 46.96% incompatible, and 16.81% diversion. It covered nine reference months; top-stock share was 30.90%. First-session match was 83.33% with 92.96% payoff remaining. Its Holm-adjusted descriptive p-value was 1.0.

## 27. Control-orientation anatomy

`cycle_04|state_2` had only one historical pair episode, failed the frozen occurrence-support gate, had a 49.55% top-stock share, target-pair C/R/L of +9.18/+31.69/0.00 bps, and T0 F0/F10 of -32.79/-42.76 bps. `cycle_07|state_6` had 20 pair episodes and target-pair C/R/L of +3.45/+23.37/+46.69 bps, but its T0 reference was highly stock-concentrated (75.90% top share), with F0/F10 -33.71/-43.75 bps. Its two-sided nonzero-payoff test passed Holm (`q=0.0099`) but is not a directional edge test and does not repair the concentration or T0-control result. The named orientations remain descriptively stronger under frozen T0 execution attribution, but the controls' uneven support prevents a universal family claim.

## 28. Loop burst and recurrence interpretation

Earlier burst work found a sharp same-orientation recurrence lift after one recent completion followed by plateau/decay. This experiment separates that frequency channel from payoff efficiency: the median multi-loop leader occurrence share was 29.41% while payoff share was 64.99%. Thus recurrence frequency cannot by itself explain typical leader dominance. Exact recurrence phase on the primary pair panel was unavailable and was not reconstructed from an unmatched population.

## 29. Route-topology interpretation

Across 1,031 route rows that fell inside frozen pair episodes, topology counts were 261 exact completions, 51 expected-leg partials, 200 expected-leg diversions, 495 incompatible first transitions, and 24 no-transition paths. Mean fixed-close payoff was +4.14, +0.18, +13.50, +84.93, and +542.59 bps respectively. The outcome tape also separates payoff realised through the route-event next open from the subsequent increment to frozen close; those means were -12.60/+16.73 for exact completion, -24.86/+38.36 for diversion, and +40.91/+44.02 for incompatible transitions. These realised outcomes differ materially by family and do not make route correctness universally economic. Every join is episode-specific and outcome-only; no topology was promoted into a causal feature.

## 30. Sequential path deterioration

The episode/phase join retained 3,592 final-leader, 1,658 non-leading-positive, and 1,221 negative-or-neutral diagnostic rows. Negative-tail rates were 45.88%, 36.85%, and 72.07%; mean frozen remaining payoff was +47.37, +88.04, and -122.93 bps. For final leaders the negative tail was 49.24% at onset, 42.87% in the middle, 46.36% late, and 44.14% at decay, so no consistent deterioration rise preceded decay. The raw one-to-three-bar score itself was not retained; only frozen path classes and outcome tails are available. MFE and MAE remain outcome-only and never enter causal indicator tables.

## 31. Indicator manifestations

Only matched, source-timestamped causal values were compared. Profitable-versus-unprofitable manifestations were stratified within each unchanged loop/current-regime pair rather than pooled across pairs. Episode middles versus pre-episode rows showed higher loop-score entropy and transition surprise, but lower top/second margin, structural breadth, and top-loop score; several changes reversed during decay. Dominant-versus-other-positive contrasts were often period-inconsistent and had substantial distribution overlap. VWAP, opening range, sector-relative strength, causal state age, and recurrence phase were unavailable on this exact panel. No universal onset signature was found and no threshold was searched.

## 32. Component-specific indicator associations

Across 117 component-association comparisons, only `positive_stock_fraction` versus the regime component passed BH FDR: pooled Spearman 0.0558 with session-block 95% interval 0.0261–0.0859 (`q=0.0392`), and 2025 Spearman 0.0796 (0.0407–0.1202; `q=0.0392`). The effect is tiny. No loop-excess indicator association survived FDR, so the indicator evidence does not identify the dominant orientation.

## 33. Common-factor diagnostic

Period-specific PCA explained 18.99%/33.97%/45.91% with one/two/three factors in 2023 and 17.94%/29.44%/40.29% in 2025. PC1 correlated with `C(t)` at 0.591/0.393; first-loading ranks correlated 0.599 across periods. Removing C and regime components slightly improved median residual lag-one stability in 2023 (+0.014) but worsened it in 2025 (-0.108). PCA never altered the primary decomposition and used no future-period loadings for earlier folds.

## 34. Co-activation network

Four edges met the frozen display support/uncertainty rule, including `cycle_04|state_4`–`cycle_07|state_6` and three other pair relationships. The complete edge table reports observed and block-null co-activation, excess, uncertainty, regime metadata, node total positive payoff, and same-regime leader frequency; the plot uses those payoff and regime attributes for node size and colour. Because the global co-activation null was not rejected and network edges are descriptive, none may select a pair.

## 35. Stock, cohort, month, and period concentration

Across all same-regime episodes, 43/107 (40.19%) breached the frozen stock/cohort concentration rule, driven largely by single-loop/sparse episodes; only 2/42 (4.76%) multi-loop episodes did. The median top-month positive-payoff share was 100%, reflecting short episodes, so episode-level calendar concentration is substantial despite coverage across months. Best-stock, top-five-stock, and every leave-one-stock-out scenario rebuild the equal-stock pair payoff panel and recalculate common, regime, and loop-excess summaries; this is row-deletion attribution, not model retraining. The retained liquidity cohort is reported where supported; exact volatility cohort, sector, and beta data are explicitly unavailable. The 2023/2025 multi-loop shares were similar, but that does not establish prospective stability.

## 36. Failure cases and blockers

The main failures are: co-activation does not exceed the persistence-aware null; Level-3 adjacency chains over-merge; one multi-loop leader share is unavailable; exact primary recurrence/state-age and several market-context fields are absent; supported sequence interactions do not replicate by the same path or survive FDR; common/regime/loop persistence is near zero; several controls are stock concentrated; and early final-leader matching is hindsight-conditioned and only 50%–60% top-one in multi-loop episodes.

## 37. What profitable periods consist of

The raw same-regime anatomy is mixed: 60.7% single-loop; among the rest, 11 extreme, 22 majority-dominant, 8 diffuse, and 1 unknown. Realised multi-loop leaders are usually payoff-efficiency-dominant. Regime contribution is slightly largest under positive-part mass, while loop excess is largest under signed positive-payoff and marginal-variance summaries. Eight episodes meet the frozen shared-regime component rule and 33 meet loop-specific activation. Broad shared-market activation is not supported after null adjustment, and sequence × loop interaction is unsupported.

## 38. Is the leader early enough for a later causal test?

Substantial payoff remains at early checkpoints, but final top-one leader agreement is only 50.0% after one session and 56.1% after the first quarter in multi-loop episodes; even at half-duration it is 59.5%. More importantly, the prerequisite that co-activation exceed the block null fails. The result is anatomically interesting but does not justify a leader selector, diversified episode basket, or sequence-conditioned prospective test from this dataset.

## 39. Scientific decision

**Primary label: `coactivation_not_above_null`.**

Optional descriptive secondary reading: realised multi-loop episodes are often `payoff_efficiency_explains_leader`, and some windows exhibit `same_regime_loop_specific_activation_dominant`. Neither secondary description overrides the primary gate. The broad registered shared-episode hypothesis is rejected on this historical population; the sequence-interaction hypothesis is not supported; no edge is confirmed.

The exact rerun compared 53 machine-readable files with no missing, extra, or hash-mismatched file; the 16 plot hashes were verified separately through each run's manifest. The independent auditor rebuilt all three episode ledgers and every panel membership, the census, block null and full co-activation network including node economics, per-pair decomposition and episode component shares, occurrence/payoff identities, leaders, every early rank/share/efficiency checkpoint, persistence statuses, all supported sequence intervals/FDR values and all four-way groups, named target-pair components, route and path phase/payoff joins, within-pair indicator identity and support/FDR, stock-removal component recalculations, concentration, PCA, scoped row traceability, artifact/plot hashes, safety flags, and primary/rerun identity; every audit check passed.

## 40. Exact next recommendation

The single most valuable next experiment is a **preregistered, read-only replication of the same eligibility- and persistence-adjusted co-activation plus occurrence/payoff-efficiency decomposition on a genuinely unopened future data snapshot**. Freeze the pair universe, positive definition, eligibility mask, five-session block null, support rules, episode adjacency, and decision gate before opening it. If co-activation again fails the null, stop the shared-episode research line. Only if it clears the null should a separate later experiment consider one causal early-anatomy hypothesis. No current loop, basket, sequence, indicator, or topology should be traded or prospectively promoted from V1.

## Predecessor evidence reconciliation

- Regime/history: consistent with magnitude/range context rather than payoff direction; the retained indicator association is tiny and regime-specific.
- Loop burst: frequency matters for some families, but median leader efficiency shows it is not the typical sole mechanism.
- Named orientations: named T0 economics remain descriptively stronger than opposite controls, with unequal concentration caveats.
- Route topology: exact completion and incompatibility retain no universal economic meaning.
- Sequential path: bad-path manifestations separate negative/neutral manifestations from positive ones, but no consistent deterioration rise precedes final-leader decay and the raw short-path score is unavailable.
- Directed rotation: the failed global excess-coactivation test gives no support to a stable family-transition chain.
- Directional atlas: no simple universal signature emerges; matched causal manifestations overlap heavily and do not identify leader excess.

## Reproducibility and safety

The frozen robust economic unit is one capped stock contribution per session × loop × orientation, equal-stock weighted, 10% winsorised, with the retained 5 bps-per-side convention; median aggregation is a frozen sensitivity. Detailed ledgers use Parquet, summaries CSV/JSON, and every analytical row includes trace identity columns. The independent safety scan confirms no broker, IG, order, position, exit, deployment, application-runtime, API-key, or secret path changed.
