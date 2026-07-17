# Fixed One-Bar Entry Latency V1

## Scientific decision

**experiment_blocked_by_missing_2023_provider_tape**

The exact 2025 post-fill latency test completed, but the registered two-period decision is blocked because none of the original 2023 provider files matched their frozen SHA-256 values. The available 2025 interpretation is **same_terminal_latency_not_supported**. This is opened retrospective research, not trading approval.

## 1. Exact hypothesis and prior boundary

The hypothesis is deliberately narrow: for the same immutable named-loop opportunity, frozen OCO direction, costs, and original terminal, T1 waits until the actual T0 breakout-fill bar completes and enters at the exact next provider open. No price sign, anchor veto, range rule, payoff-state gate, or fitted model is used.

This exact T0/T1 comparison had not previously been tested. Clean Anchor Price Acceptance V1 used `anchor+10m` for every opportunity. The source T0 is instead `anchor + 5 * entry_step` and `entry_step` ranges from 1 to 24. In 2025, 206 of 809 named rows have `entry_step > 1`; on those rows the Clean Anchor clock was not one bar after T0 and could precede the causal breakout direction. Its +27,936.20 bps level is therefore context, not the registered latency result.

## 2. Scientific status, frozen populations, and controls

- `cycle_04|state_4`: 2023=132, 2025=96.
- `cycle_07|state_5`: 2023=722, 2025=713.
- Controls remain separate: `cycle_04|state_2` and `cycle_07|state_6`.
- No failed loop is replaced; overlap and capacity are never refilled.

## 3. Source identity and exact clocks

T0 is the stored V2 `no_payoff_state_filter` OCO fill. Its timestamp is the start of the trigger bar; its price is the frozen breakout threshold or opening gap fill, not necessarily the provider open. T1 is exactly `T0 timestamp + 5 minutes`, at that provider bar's open. T2 is `T0 + 10 minutes` and remains a secondary shape diagnostic. All primary rows retain the stored terminal `anchor + 125 minutes`, priced at the close of the provider bar beginning five minutes earlier. Entry and exit each cost 5 bps.

The exact source and delayed rows are paired by opportunity, anchor, event lineage, symbol, session, loop, orientation, direction, and terminal. Missing T1 rows remain explicit; T0 metrics below are recomputed only on exact pairs.

## 4. 2023 archival status and paired population

The expired root `/private/tmp/stocker_eodhd_pre2024_intraday_20260710/source=eodhd/instrument_type=stock` does not exist. The pre-score archival search hashed 20 candidate files and found 0 matches across 20 required symbols. No fresh download, approximate field, or imputation was used. All 854 named 2023 T1 outcomes remain missing.

Across both frozen source periods there are 1663 named opportunities and 1663 exact stored T0 rows. The immutable provider evidence yields 808 exact T1 rows and 808 pairs (48.6%); every missing 2023 T1 remains explicit rather than becoming a zero.

## 5. Primary T0, T1, and paired result

- T0: 15087.32 bps total, 18.67 bps per pair.
- T1: 6909.77 bps total, 8.55 bps per pair.
- T1 minus T0: **-8177.55 bps**, mean -10.12, median -4.76.
- Opportunities improved: 45.5%; sessions improved: 41.1%.
- Five-session-block 95% interval for the session-mean delta: [-18.54, -3.29] bps.

## 6. Named-loop results

| named loop | pairs | T0 net bps | T1 net bps | delta bps | mean delta |
|---|---:|---:|---:|---:|---:|
| cycle_04 | 96 | 2951.93 | 2841.00 | -110.93 | -1.16 |
| cycle_07 | 712 | 12135.39 | 4068.77 | -8066.61 | -11.33 |

## 7. Period and direction results

| period | source rows | exact pairs | T0 paired net bps | T1 net bps | delta bps |
|---|---:|---:|---:|---:|---:|
| 2023 | 854 | 0 | unavailable | unavailable | unavailable |
| 2025 | 809 | 808 | 15087.32 | 6909.77 | -8177.55 |

| direction | pairs | T0 net bps | T1 net bps | delta bps |
|---|---:|---:|---:|---:|
| long | 550 | 13652.60 | 5812.23 | -7840.36 |
| short | 258 | 1434.72 | 1097.54 | -337.19 |

The 2023 row is unavailable by construction and is never interpreted as no effect. The 2025 row is the complete exact archival result currently available.

## 8. Frozen control results

| control orientation | pairs | T0 net bps | T1 net bps | delta bps | mean delta |
|---|---:|---:|---:|---:|---:|
| state_2 | 6 | -196.76 | -265.54 | -68.78 | -11.46 |
| state_6 | 296 | -9977.35 | -10396.07 | -418.72 | -1.41 |

The named-versus-control comparison determines whether latency is loop-specific or a general execution-clock effect; controls are never pooled into the primary endpoint.

## 9. Entry-price decomposition

The mean direction-adjusted T0-to-T1 entry move is 10.18 bps and the median is 4.73 bps. Negative values mean price moved against the frozen direction and offered a better delayed price. Such adverse moves occur in 368 of 808 pairs (45.5%). Among T0-profitable rows, 148 of 404 (36.6%) move adversely before T1. The entry-move/delta Spearman relationship is -1.000; the exact return-convention reconciliation error is checked row by row and audited independently.

This decomposition is diagnostic only. It does not create an inverse acceptance rule or select a subset.

## 10. Costs, T2, and restarted horizon

At frozen costs, paired T0 and T1 levels are 15087.32 and 6909.77 bps, with delta -8177.55. At twice costs they are 7007.32 and -1170.23, with unchanged delta -8177.55 because this frozen model charges identical fixed-bps entry and exit costs.

T2 has 807 exact pairs and leaves a mean 21.29 bars (minimum 2) before the original terminal; T2 minus T0 is -5990.60 bps and T2 minus T1 on the common population is 2170.07 bps. T2 cannot replace the T1 endpoint. Restarted-h24 outcomes are exported separately and never enter the same-terminal conclusion.

## 11. Concentration, deletions, and leave-one-stock-out

The largest stock contributes 16.7% of absolute paired delta and the top five contribute 56.9%; stock HHI is 0.090.

- remove_best_stock: 768 pairs, delta -8758.00 bps.
- remove_top_five_stocks: 613 pairs, delta -9547.11 bps.
- remove_best_episode: 790 pairs, delta -8398.51 bps.
- remove_top_five_episodes: 751 pairs, delta -8516.08 bps.

Because the latency rule has no trained or stock-dependent state, a conventional model rebuild is not applicable. `leave_one_stock_out_results.csv` exactly recomputes every remaining deterministic pair after excluding each stock and labels this honestly rather than calling row deletion a trained-model rebuild.

## 12. Failure cases and interpretation

The main failure modes are missing exact T1 opens, T1 at or after the original terminal, absent 2023 provider evidence, period heterogeneity, control replication, and concentration. A positive standalone T1 level is not incremental evidence; only paired T1-minus-T0 counts. Hindsight episodes appear only in concentration and attribution outputs.

The available evidence is classified as **same_terminal_latency_not_supported**. The formal result remains **experiment_blocked_by_missing_2023_provider_tape** because success required both 2023 and 2025 plus prospective confirmation.

## 13. Exact recommendation

The single most valuable next step is an execution-free prospective cohort using the immutable T0 opportunity ledger, exact create-only T1 timing record, and later separate outcome settlement. Do not deploy or tune the delay. If an original 2023 tape matching the registered per-symbol hashes is recovered, rerun this unchanged contract to close the archival two-period question.

## Reproducibility

- Run ID: `fixed-latency-d66f96ac8f08d9c8293d14e7`
- Git SHA: `1e0e1a4149961589b3213940d56fb6dd9450565d`
- Contract SHA-256: `35fa576fd87acc85220105807beafdc3103db87fbf3c0a7dd0044afe4877ff84`
- Data snapshot SHA-256: `08752c75f98f2c61114496ebf78eab2c055adeb138a72c88bb1d1981cf72a96a`
- Command: `PYTHONPATH=packages/stocker_research/src .venv/bin/python research/slrno-v2/20260714-regime-loop-handoff/work/run_fixed_one_bar_entry_latency_v1.py --output <OUTPUT> --report <REPORT>`

## Validation

- 41 focused fixed-latency tests passed.
- 226 relevant V1/V2/lead-lag/sequential-veto/rotation/clean-anchor/fixed-latency tests passed.
- The full repository suite passed: 595 tests.
- Scoped Ruff format/lint passed; strict mypy passed for all six new runner, auditor, and reusable-module source files.
- Primary versus exact rerun passed byte identity across 39 machine-readable files and plots.
- The independent auditor passed all 14 checks, including 1,110 raw provider reconstructions (808 named), all paired metrics, the block interval, costs, nulls, concentration, T2, missing-data accounting, and safety boundaries.
- Repository-wide Ruff still reports 1,153 pre-existing unrelated errors; none are in the scoped fixed-latency files.
- `git diff --check` passed.

## Non-result-driven implementation corrections

Before scoring, two-axis review removed an over-strict intermediate-path gate, corrected adverse-excursion signs, fixed episode deletion and the prior-session null, and broadened independent audit coverage. The first scoring attempt then exposed a report-only paired-count column collision; commit `856447ff8373b4a478041b20522c690cf868d64d` fixed it and all outputs were regenerated. The first audit failed closed on `<NA>` versus CSV `NaN` contributor-key representation; that auditor-only defect was corrected, tested, and re-audited. No candidate, latency, threshold, direction, terminal, cost, population, or economic result was changed in response to P&L.
