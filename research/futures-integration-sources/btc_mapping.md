FAST_BTC_REAL_EXPIRY_RECONSTRUCTION_V0 · DEVELOPMENT_DIAGNOSTIC

1. **Closest product depends on entry time.** The frozen intent is a bullish, approximately 10-delta same-day call on standard BTC futures. The mechanical mapping selects the earliest still-live same-day expiry: standard Bitcoin options before 11:00 New York in this sample; BFF options from 11:00 until before 16:00. Micro options offer smaller size but the same earlier expiry. [CME option rules](https://www.cmegroup.com/cryptooptionsfaq)
2. **299/341 purchases have a rules-based same-day product/time counterpart; zero exact historical option contracts are verified.** These comprise 86 standard BTC and 213 BFF candidates. All 42 purchases at 16:00 are NO_REAL_0DTE_MATCH. Historical chains/strikes are unavailable; 135 nearest-grid model candidates can be priced at entry, 91 at minute60 and 33 at estimated expiry. The 27 original nonpurchases remain listed.
3. **The inherited 17:00 expiry is incorrect for these products.** Standard/Micro expiry is 16:00 London (11:00 NY here); BFF expiry is 16:00 NY. No mapped same-day option has life remaining after the synthetic 17:00 boundary. The source is **C: purely synthetic expiry architecture**, not a replicated listed contract. [Standard rulebook](https://www.cmegroup.com/content/dam/cmegroup/rulebook/CME/IV/350/350A.pdf), [BFF rulebook](https://www.cmegroup.com/rulebook/CME/V/450/451A/451A.pdf)
4. **Current synthetic loss rate: 246/341 = 72.14%.** All 341 purchase outcomes reconcile exactly; there are 75 near-total losses (≥£90), 42 approximately full-premium losses (≥£99.99), gross losses £16,073.22, gross profits £24,090.77 and net +£8,017.55. Principal/+1000 attainment counts are 33/9.
5. **A verified whole-sample real-expiry loss rate is unavailable.** Conditional model results are 74/91 = 81.32% at minute60 and 32/33 = 96.97% at estimated expiry. Same-identity synthetic rates are 61.54% and 63.64%, respectively. The subsets differ; neither can replace the 341-trade control rate.
6. **Near-total losses do not broadly disappear in the assessed subsets.** At minute60, eight matched synthetic near-total losses become seven retained plus one avoided, with 24 new near-total losses: 31/91 total, +25.27 percentage points versus matched control. At estimated expiry, four remain and 28 new losses appear: 32/33, +84.85 points. Unassessed trades are not treated as saved losses.
7. **16:00 is a mapping failure, not evidence of improved real P&L.** Its synthetic outcome is 41/42 losers and near-total losses, net −£2,006.69. At entry the latest same-day BFF contract has already terminated; tomorrow is not substituted. At 15:00 only six model outcomes are assessable, all full-premium losses; the 42-trade synthetic cohort loses 76.19%, versus 83.33% on those six matched controls.
8. **Minute60 remains a fixed elapsed-time research endpoint, not a universal expiry.** It precedes expiry for 09:00 and 11:00–14:00 mappings, and coincides with expiry for 10:00/15:00. Missing fixing inputs at that endpoint remain unavailable. These same-day mappings do not cross midnight or weekends. The 42 synthetic 16:00 purchases reach the old 17:00 boundary and current maintenance start, but have no corresponding real option.
9. **REAL_OPTION_MARKS_UNAVAILABLE; EXECUTION_ECONOMICS_UNVERIFIED.** Ten weekly BFF futures histories were obtained through the existing approved read-only research source; these are not option quotes. No midpoint or executable bid/ask P&L is claimed. The source does not provide expired option/FOP history, and no such marks were cached. [IB historical-data limits](https://interactivebrokers.github.io/tws-api/historical_limitations.html)
10. **BTC's contribution after correction is unknown.** In the saved, unrecomputed overall reference, BTC supplies 246 of 1,458 losers (16.87%) and 11.83 percentage points of the 70.10% loss rate across 2,080 known outcomes. Removing unmatched purchases would change the entry method, so no corrected pooled rate is manufactured.

**Primary conclusion: INCONCLUSIVE_REAL_BTC_MAPPING.** The inherited expiry architecture is invalid as a listed-contract replication. The available conditional calculations do not support a claim that correcting expiry reduces BTC losses, but contract verification and data coverage are insufficient for a definitive economic verdict.

| Evidence / endpoint | EARLY N / loss rate | LATE N / loss rate | ALL N / loss rate | ALL P&L |
|---|---:|---:|---:|---:|
| CONTROL_SYNTHETIC | 168 / 71.43% | 173 / 72.83% | 341 / 72.14% | £8,017.55 |
| Conditional MODEL_REAL_EXPIRY minute60 | 42 / 85.71% | 49 / 77.55% | 91 / 81.32% | £45,177.83 |
| Conditional MODEL_REAL_EXPIRY estimated expiry | 15 / 100.00% | 18 / 94.44% | 33 / 96.97% | −£2,630.01 |
| Actual historical option marks | 0 / unavailable | 0 / unavailable | 0 / unavailable | unavailable |

Minute60 matched-control loss-rate changes are +26.19 points EARLY / +14.29 LATE; estimated-expiry changes are +40.00 / +27.78. The large minute60 model profit is dominated by two 09:00 trades (+£31,995.78 and +£12,932.34), not a fall in losing frequency. Tiny model premiums and coarse strikes make it unsuitable as an executable-profit estimate.

**Model and delta limits.** Source entry uses the completed prior-minute futures close; Black76 has r=0, ACT/365, sigma from 15 completed log returns, and a continuous inverse-10-delta strike. Sigma stays fixed forward. It assigns same-date 17:00 NY expiry and intrinsic-only value at expiry, with no documented listed fixing. Reconstruction preserves this pricing/volatility methodology while using the required product's underlying, documented expiry, and nearest model-delta strike on the published lattice. Thus futures basis, strike discretisation and causal volatility inputs can also differ; this is not a pure expiry-only causal estimate. The 135 priceable candidates have estimated deltas from 0.000464 to 0.179491 against target 0.10. No tolerance was invented to label those all good matches. For 163 BFF entries the unchanged RV formula is zero; one more lacks completed prior bars. They remain unassessed without adding a volatility floor.

Standard/Micro same-day strike rules changed on June 1, 2026; the audit uses the $250 short-dated lattice applicable below $100,000, not older intervals. BFF near-term rules use $100 above $50,000. Historical listing-band anchors and exact chains remain unverified. [June 2026 strike amendment](https://www.cmegroup.com/content/dam/cmegroup/notices/clearing/2026/05/chadv26-187.pdf), [BFF strike amendment](https://www.cmegroup.com/content/dam/cmegroup/market-regulation/rule-filings/2025/7/25-289.pdf)

**Fixing and schedule.** BFF Monday–Thursday expiry values are explicitly MODEL_ESTIMATED_FIXING_PROVIDER_VWAP_NOT_OFFICIAL: complete 15:00–16:00 NY minute-bar trade averages weighted by volume, then cash intrinsic. Provider filtering may differ from CME's fixing. Friday BRRNY, monthly BRR, and standard weekly combined BTC/Micro fixing inputs are absent; no closing-price substitute is used. Black pricing before fixing remains an approximation to a fixing-settled product. [CME fixing mechanics](https://www.cmegroup.com/cryptooptionsfaq)

The entire July 24–September 25 sample is after the May 29, 2026 expansion. Weekday maintenance is 16:00–16:02 CT; Saturday maintenance is normally 02:00–04:00 CT. Trading hours, option expiry and fixing windows are separate fields in the rules audit. No new weekend entries or next-day expiry substitutions were added. [Dated schedule notice](https://www.cmegroup.com/notices/electronic-trading/2026/05/20260525.html), [CME maintenance schedule](https://cmegroupclientsite.atlassian.net/wiki/spaces/EPICSANDBOX/pages/1283194884)

**Artifacts and scope.** The comparison CSV contains all 368 identities plus EARLY/LATE/ALL summaries; the mapping CSV contains per-entry timestamps, expiry/fixing, delta/strike/premium and 16:00 sanity fields. Clock and coverage files retain all denominators. CME_RULES_AUDIT.json records three-product rules and dated references. QA.json records reconciliation, hashes and missing-data limitations. The final “YES” below refers to the conditional research model's documented rules; it does not certify a complete historical listed-contract reconstruction. Research only; ordering disabled; no production changes. STOP.

BTC entry method changed: NO

BTC direction changed: NO

BTC target delta changed: NO

New filters tested: ZERO

New exits tested: ZERO

Expiry/trading model corrected to listed contract rules: YES

Executable option pricing verified: NO
