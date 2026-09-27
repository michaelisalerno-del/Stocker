FAST_GC16_G2_DECONFOUND_V0 · DEVELOPMENT_DIAGNOSTIC

1. **26 GC frozen-trigger events occur at exactly 16:00 New York:** 12 EARLY / 14 LATE; G2 selects 10 / 9.
2. **GC16 HOLD_ALL: £-2,273.52** (£-873.52 EARLY; £-1,400.00 LATE).
3. **EXIT_ALL: £-449.01; +£1,824.51 versus hold.** It exits 25 helpful trades and one harmful trade; tail details below.
4. **Frozen G2: £-592.44; +£1,681.09 versus hold.** All 19 selected exits help; no false exits. Improvement is +£919.03 / +£762.05.
5. **G2_SELECTED does not consistently outperform G2_NOT_SELECTED inside GC16.** Mean exit advantage is £91.90 versus −£141.04 EARLY, but £84.67 versus £85.10 LATE. LATE unselected exits all help.
6. **Positive G2 improvement repeats; incremental selection information does not.** EARLY G2 beats EXIT_ALL by £282.08, but loses £425.51 of available improvement LATE. Pooled EXIT_ALL earns £143.43 more.
7. **Outside 16:00:** G2 adds £-8.70 EARLY / £59.82 LATE; £51.11 overall. The requested negative-control sign pattern is absent.
8. **Date/trade deletion survives in both halves.** See the compact deletion table below; no reselection.
9. **All GC +1000 trigger opportunities are preserved by G2: 2/2**, including the sole GC16 giant. Retention means attainment before policy exit, not cash realised at the maximum.
10. **IV1.25 preserves relative improvement:** +£738.32 / +£545.20. **Candidate absolute P&L is negative in both halves:** £-98.55 / £-454.80.

**Primary verdict: GC16_CLOCK_EFFECT_BUT_G2_ADDS_LITTLE.** The clock cohort accounts for 97.0% of GC G2 improvement. A strong conditional-information claim fails: the selected-versus-unselected mean relationship does not repeat in LATE, and EARLY has only two unselected events. G2 avoids one damaging EARLY exit, so the evidence is more nuanced than a pure clock proxy. EXIT_ALL is not promoted.

**Source reconciliation and freeze.** Exact GC improvement is £910.331923429762680000 EARLY + £821.8685844218982090000 LATE = £1732.2005078516608890000. All 117 trigger identities and 25 selections match; 116 endpoints are known and 24 selected endpoints are known. The selected unknown is GC|2026-09-07|462941472|840 (14:00), excluded from paired P&L. CURRENT_RESEARCH_NG13_ONLY lineage, put_10d_0DTE, original purchases/direction/strikes/expiry, £100 normalisation, minute60, local partitions, +50% arm, breakeven trigger and next-minute fills are unchanged. G2 remains underlying_range3 <= 0.0004699192926821846.

| Half / group | N | Help / hurt / zero | Mean advantage | Median | Total advantage |
|---|---|---|---|---|---|
| EARLY / selected | 10 | 10 / 0 / 0 | £91.90 | £93.04 | £919.03 |
| EARLY / not selected | 2 | 1 / 1 / 0 | £-141.04 | £-141.04 | £-282.08 |
| LATE / selected | 9 | 9 / 0 / 0 | £84.67 | £90.91 | £762.05 |
| LATE / not selected | 5 | 5 / 0 / 0 | £85.10 | £81.76 | £425.51 |
| ALL / selected | 19 | 19 / 0 / 0 | £88.48 | £91.86 | £1,681.09 |
| ALL / not selected | 7 | 6 / 1 / 0 | £20.49 | £72.88 | £143.43 |

| Half | After best date | After best trade | After best two trades |
|---|---|---|---|
| EARLY | £794.60 | £794.60 | £678.41 |
| LATE | £649.04 | £649.04 | £553.40 |

**Concentration.** Largest positive date: 2026-07-27, £124.43, 7.40% of net improvement. No negative selected date. Top two dates: 2026-07-27 (£124.43), 2026-08-04 (£116.19); together 14.31%. Top five trades contribute 32.73%. Every selected GC16 event has a different date. EARLY/LATE best-date shares are 13.54% / 14.83%. Date and best-trade deletion therefore coincide.

**Tails.** GC16 has 3 principal (+500%) and 1 giant (+1000%) opportunities; G2 retains all, holds all three tail trades, and causes zero tail deterioration. Principal endpoint/candidate totals are £26.48 / £26.48; giant totals £226.48 / £226.48. EXIT_ALL retains 2/3 principal and 0/1 giant opportunities. Its one false exit costs £299.92 on 2026-07-30. The CSV separates already-attained tails from first attainment after exit; maxima never determine fills.

**Feature effect.** GC16 Spearman(range3, exit advantage) is -0.469 / -0.398 EARLY/LATE. There is only one EXIT_HURT event EARLY and none LATE; class-specific correlations are consequently unassessable for hurt events. Means, medians, quartiles, valid correlations and only the frozen two-way cut are in ROBUSTNESS.csv.

**Stress and limits.** The saved stress framework is compatible. All 44 original GC16 purchases are replayed at sigma ×1.25 on original IV1.00 strikes and expiry; scenario entry premiums, arms, triggers and range3 are independently recomputed. There are 20 stressed trigger events (17 selected). No stressed GC16 principal/giant opportunity exists, so stress tail retention is vacuous. Primary GC16 G2 P&L is positive EARLY (£45.51), negative LATE (−£637.95); relative benefit is not absolute profitability. Small, previously exposed samples and the synthetic expiry prevent an independent-validation claim.

**Files and QA.** GC16_EVENT_COMPARISON.csv contains all 26 primary and 20 stress GC16 events. GC_CLOCK_SPLIT.csv contains both primary clock partitions in EARLY/LATE/ALL. GC16_POLICY_COMPARISON.csv contains only HOLD_ALL, EXIT_ALL and G2, including stress. ROBUSTNESS.csv contains group economics, deletions, feature descriptions, all requested contribution dimensions and top-five trades. QA.json records full-precision reconciliation, all GC source identities, source hashes, metric definitions, concentration and checks. All 117 primary GC trigger paths and the stress GC16 purchase universe are replay-verified. Research only; live ordering disabled; order placement disabled. Cached local files only; no downloads, external calls, production changes or orders. STOP.

New thresholds discovered: ZERO

New features discovered: ZERO

Frozen G2 threshold changed: NO

Market tested: GC only
