# Minimal Intraday Stock → IV-Excess Holdout Validation V0.1

Overall decision: `blocked_insufficient_frozen_tail_support`.

The minimal M1 forecast increment transferred to the untouched holdout and passed the frozen model gate. The frozen M1 tail had positive mean and median IV residuals and an exceed-IV rate above 50%, but it did not pass the preregistered tail gate: one stock supplied 18.2336% of tail rows, above the 18% ceiling, and the coarse 80% bootstrap lower bound for the median residual was negative. The tail result is therefore not validated; it is not evidence of option profitability, prospective trading utility, or a deployable strategy.

## Frozen design and chronology

- Training: 2024-01-01 through 2024-12-31.
- Historical reference: 2025-01-01 through 2025-08-22, reconstruction only and not used for tuning.
- Binding holdout: 2025-09-01 through 2025-12-31.
- M0: frozen Group O, previous-close front-options context.
- M1: frozen Group O plus Group I, intraday H0 stock condition.
- Excluded: daily stock features/regimes, route competition, route-resolution state, and hand-built mismatch features.
- Historical reconstruction passed. M0 reproduced G0/C0 with zero row, probability, metric, coefficient, and threshold differences; M1 remained exactly Group O + Group I.
- Frozen weighted 2024 thresholds: M0 `0.4443768830968764`; M1 `0.49588519865576763`.
- Outcomes were opened only after coverage preflight and the model/threshold freeze.

## Resume acquisition and coverage

- Complete V0 receipts found/reused/corrupt: 1,450/1,450/0; redundant network requests prevented: 1,450.
- Missing logical requests at resume: 250; attempted/completed/failed: 250/250/0; remaining: 0.
- Interrupted SMCI request for signal session 2025-09-09 and required observation date 2025-09-08 was redownloaded from its beginning. The 198-row incomplete page contributed zero admitted rows; the completed response contained 263 records and 228 exact-date records.
- New provider records/exact-date records/bytes: 57,426/49,584/52,338,919.
- New extra-date/2026-or-later records rejected: 7,842/193; protected or unauthorised rows materialised: 0.
- Cumulative provider records: 407,426, including the excluded 198-row orphan page; completed-receipt records: 407,228.
- The independent receipt rebuild reloaded all 1,700 receipts with zero network requests, rejected 68,946 extra-date records including 2,026 protected-date records, and reproduced all 261,298 canonical rows byte-for-byte.
- Planned holdout universe: 1,700 stock-sessions across 85 US trading sessions and 20 stocks.
- Exact previous-close chains/valid frozen ATM pairs: 1,313/1,214. All 80 planned stock-month cells were represented.
- Coverage preflight passed before outcomes: 13,070 expected joined rows, 68 sessions, 20 stocks, all four months; maximum weighted stock/month shares 5.5441%/28.0018%.

## Holdout model results

| Metric | M0 | M1 | M1−M0 improvement |
|---|---:|---:|---:|
| Log loss | 0.58350936 | 0.57234591 | 0.01116345 |
| Brier score | 0.19791099 | 0.19334631 | 0.00456468 |
| AUC | 0.61043506 | 0.65409618 | 0.04366112 |
| Average precision | 0.36907997 | 0.41438403 | 0.04530406 |
| Expected calibration error | 0.02864515 | 0.04613967 | -0.01749452 |
| Calibration intercept | 0.04455142 | -0.08558864 | absolute-error change -0.04103723 |
| Calibration slope | 1.24250929 | 1.21661293 | absolute-error improvement 0.02589636 |
| Mean realised-class probability | 0.59190534 | 0.59551958 | 0.00361425 |

Both models were scored on 13,070 rows, 68 sessions, 20 stocks, four months, and 3,658 positive outcomes. The weighted base rate was 0.28352869. Joined support passed every fixed gate.

## Monthly stability

| Month | M0 log loss | M1 log loss | Δ log loss | Δ Brier | Δ AUC | Δ AP | M1-tail mean residual | Median residual | Exceed rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2025-09 | 0.57541439 | 0.55815925 | 0.01725513 | 0.00744204 | 0.04703488 | 0.06898185 | 0.00400793 | 0.00350097 | 0.595325 |
| 2025-10 | 0.56079127 | 0.54976402 | 0.01102726 | 0.00431611 | 0.06385856 | 0.07097816 | 0.00240722 | -0.00159771 | 0.430788 |
| 2025-11 | 0.61566597 | 0.60452318 | 0.01114280 | 0.00480151 | 0.03631534 | 0.03356112 | 0.00492480 | 0.00139887 | 0.524748 |
| 2025-12 | 0.58790921 | 0.58240834 | 0.00550087 | 0.00187539 | 0.01604300 | -0.00015766 | 0.00331501 | 0.00053933 | 0.529278 |

Log-loss improvement was positive in all four months. Tail mean residual was positive in all four months and tail median residual in three.

## Checkpoint and frozen-subgroup stability

| Group | M0 log loss | M1 log loss | Δ log loss | Δ Brier | Δ AUC | Δ AP | M1-tail mean residual | Median residual | Exceed rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Early checkpoints 6–14 | 0.65051238 | 0.63723653 | 0.01327585 | 0.00606284 | 0.06305798 | 0.05831763 | 0.00392412 | 0.00082639 | 0.526513 |
| Middle checkpoints 16–24 | 0.56718641 | 0.55248064 | 0.01470577 | 0.00588808 | 0.10004900 | 0.08660000 | 0.00297195 | -0.00006971 | 0.483316 |
| Late checkpoints 26–34 | 0.49621355 | 0.49183739 | 0.00437616 | 0.00093347 | 0.08251739 | 0.05265151 | 0.00183947 | -0.00109015 | 0.334825 |
| Low prior-close ATM IV | 0.57638763 | 0.56283756 | 0.01355007 | 0.00558809 | 0.03990778 | 0.05107687 | 0.00353583 | 0.00096428 | 0.542367 |
| High prior-close ATM IV | 0.59213204 | 0.58385820 | 0.00827384 | 0.00332557 | 0.03771742 | 0.02411981 | 0.00380022 | -0.00054422 | 0.486000 |
| Low H0 transition probability | 0.61206640 | 0.60472586 | 0.00734054 | 0.00325166 | 0.05117919 | 0.05951367 | 0.00358620 | 0.00025472 | 0.504107 |
| High H0 transition probability | 0.55329604 | 0.53808795 | 0.01520809 | 0.00595385 | 0.03389909 | 0.03181015 | 0.00437310 | 0.00160291 | 0.560973 |

No checkpoint group was materially adverse under the frozen model gate.

## Frozen M1 top-5% tail

- Support: 702 rows, 65 sessions, 20 stocks, four months.
- Mean/median absolute 15-minute movement: 0.01361837/0.00983902.
- Mean IV expectation: 0.00993576.
- Mean/median/10%-trimmed mean IV residual: 0.00368261/0.00053409/0.00206348.
- Exceed-IV and positive-residual rate: 0.511074.
- IV sigma ratio: 1.093614.
- Residual p05/p25/p75/p95: -0.00928348/-0.00462045/0.01004964/0.02387870.
- Largest 5% of rows contributed 27.5819% of total positive residual.
- Maximum stock/month/session shares: 18.2336%/35.4701%/5.4131%.

The row, session, stock, month, month-share, and session-share gates passed. The stock-share gate failed because 18.2336% exceeded the frozen 18% maximum. The 80% bootstrap lower bound for median residual was also negative.

## Frozen M0 versus M1 tails

M0 selected 214 rows across 60 sessions, 20 stocks, and four months. Its mean/median residual and exceed rate were 0.00066359/-0.00129307/0.413652.

| M1−M0 tail comparison | Difference |
|---|---:|
| Mean IV residual | 0.00301902 |
| Median IV residual | 0.00182716 |
| Exceed-IV rate | 0.09742207 |
| Absolute movement | 0.00567376 |
| IV sigma ratio | 0.22301081 |
| Positive-residual rate | 0.09742207 |
| Largest-row concentration | 0.00488129 |

Tail intersection/union/Jaccard were 91/825/0.110303. M1-only rows numbered 611 and M0-only rows 123.

## Movement timing for the frozen M1 tail

| Horizon | Mean residual | Median residual | Exceed rate | Fraction of 30m movement realised | Maximum-excursion bucket share |
|---|---:|---:|---:|---:|---:|
| 5m | 0.00300903 | 0.00127580 | 0.580950 | 0.479547 | 0.092519 |
| 10m | 0.00356080 | 0.00180647 | 0.613451 | 0.640094 | 0.164603 |
| 15m | 0.00368261 | 0.00053409 | 0.511074 | 0.746749 | 0.233807 |
| 30m | 0.00418559 | 0.00162470 | 0.541768 | 1.000000 | 0.509071 |

All 702 M1-tail rows had every timing horizon. One non-tail joined row lacked an optional sixth future bar; no binding 5-, 10-, or 15-minute outcome was missing.

## Ten-draw whole-session bootstrap intervals

| Statistic | 80% interval | 90% interval | 95% interval |
|---|---:|---:|---:|
| Δ log loss | [0.00817751, 0.01453867] | [0.00811354, 0.01467575] | [0.00808155, 0.01474429] |
| Δ Brier | [0.00319779, 0.00604503] | [0.00317893, 0.00604866] | [0.00316950, 0.00605048] |
| Δ AUC | [0.03623492, 0.05129925] | [0.03397384, 0.05165767] | [0.03284330, 0.05183687] |
| Δ average precision | [0.03532179, 0.05586746] | [0.03511467, 0.05882273] | [0.03501111, 0.06030036] |
| M1-tail mean residual | [0.00286689, 0.00377345] | [0.00238346, 0.00413241] | [0.00214175, 0.00431189] |
| M1-tail median residual | [-0.00061550, 0.00051185] | [-0.00069080, 0.00061972] | [-0.00072844, 0.00067365] |
| M1-tail exceed rate | [0.47424365, 0.51039964] | [0.46919237, 0.51389399] | [0.46666672, 0.51564117] |
| M1−M0 tail mean residual | [0.00169105, 0.00362685] | [0.00147329, 0.00393399] | [0.00136441, 0.00408756] |
| M1−M0 tail median residual | [0.00051516, 0.00270676] | [0.00047912, 0.00281087] | [0.00046111, 0.00286293] |
| M1−M0 tail exceed rate | [0.05319284, 0.16335605] | [0.02819740, 0.18720515] | [0.01569969, 0.19912970] |

These ten draws are a coarse diagnostic.

## Intraday-H0 bundle null

The real M1 increment exceeded all three fixed-seed Group-I bundle null increments for log loss, Brier score, AUC, and average precision. No precise p-value is inferred.

## Decision and audit

- `minimal_model_status`: `supported`
- `frozen_top_5pct_status`: `insufficient_support`
- `options_only_tail_comparison_status`: `insufficient_support`
- `movement_timing_status`: `insufficient_support`
- `holdout_options_coverage_status`: `supported`
- `download_resume_status`: `supported`
- Independent audit: passed, including manual probability reconstruction on 100 rows per model.
- Determinism: passed with zero selected-contract, joined-row, probability, feature, tail-membership, and movement mismatches.
- The strengthened audit independently rebuilt canonical data from all receipts and reconstructed selected pairs from that rebuild. Bootstrap evidence was recalculated from stored session multiplicities without redrawing; null evidence was reconstructed without refitting.

The binding 15-minute point estimates were positive, so the earlier tail did not simply disappear. However, the fixed support and bootstrap-median requirements failed. The defensible result is therefore: the minimal model forecast increment transferred, while the frozen IV-excess tail remains unvalidated.
