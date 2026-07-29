# High-Movement Pressure-Onset Screen V0.1 — Support-Semantics Repair

**Decision:** `no_pressure_onset_increment`

Scientific status: `opened_support_contract_repair_retrospective_feasibility_evidence`.

This is retrospective, research-only, observable-only feasibility evidence. It is not prospective validation, a strategy, achieved P&L, or executable edge.

## Repair integrity

- The 15-stock support gate is applied to the parent fixed-clock slate before movement admission.
- Singleton admitted slates are retained and receive row weight `1.0`.
- Concentration is evaluated after fitting and cannot hide model results.
- Frozen V0 panel identity passed across `15093` rows and `239` pre-fit columns.
- Frozen predecessor reconstruction passed: `True`; maximum probability error `0`.
- Protected rows materialised: `0`.

## Population

- Primary rows / sessions / stocks: `1560` / `153` / `20`.
- UP / DOWN / NO_ONSET: `345` / `336` / `879`.
- Assessment parent-stock distribution: `{13: 2, 15: 2, 18: 8, 19: 13, 20: 291}`.
- Assessment admitted-stock distribution: `{0: 17, 1: 33, 2: 40, 3: 35, 4: 40, 5: 31, 6: 32, 7: 21, 8: 21, 9: 13, 10: 7, 11: 11, 12: 4, 13: 6, 14: 1, 16: 2, 20: 2}`.
- Singleton / multi-candidate admitted slates: `33` / `266`.

## Onset occurrence models

| Model | Brier | Log loss | AUC | Rows |
|---|---:|---:|---:|---:|
| A0 | 0.233783736 | 0.662186839 | 0.659194219 | 1560 |
| A1 | 0.234859061 | 0.666597419 | 0.656533005 | 1560 |
| A2 | 0.233629664 | 0.661759675 | 0.660889844 | 1560 |
| A3 | 0.235522491 | 0.668012410 | 0.662079288 | 1560 |

## Direction conditional on actual onset

| Model | Brier | Log loss | AUC | Rows |
|---|---:|---:|---:|---:|
| D0 | 0.254233304 | 0.701634453 | 0.421937543 | 681 |
| D1 | 0.256024402 | 0.705765472 | 0.502665631 | 681 |
| D2 | 0.273374831 | 0.748072029 | 0.497015183 | 681 |
| D3 | 0.282973281 | 0.773360184 | 0.488354037 | 681 |

## Frozen comparisons

- `A2_minus_A1_brier`: `0.00122939627995`
- `A2_minus_A1_log_loss`: `0.00483774443215`
- `D2_minus_D1_brier`: `-0.0173504287875`
- `D2_minus_D1_log_loss`: `-0.0423065570939`
- `A3_minus_A2_brier`: `-0.00189282693197`
- `A3_minus_A2_log_loss`: `-0.00625273543842`
- `D3_minus_D2_brier`: `-0.00959845070952`
- `D3_minus_D2_log_loss`: `-0.0252881545591`

## Session-block bootstrap

- `A2_minus_A1_brier_improvement`: 90% `[-0.00221768712092, 0.00403273182607]`; 95% `[-0.00280369828767, 0.00503861046861]`.
- `A2_minus_A1_log_loss_improvement`: 90% `[-0.00361778288331, 0.0125879187287]`; 95% `[-0.00528432146685, 0.0161135626041]`.
- `A3_minus_A2_brier_improvement`: 90% `[-0.00460211347373, 0.00094535695773]`; 95% `[-0.00488764061885, 0.0013927635059]`.
- `D2_minus_D1_brier_improvement`: 90% `[-0.0247903534826, -0.00974383726398]`; 95% `[-0.0264292261806, -0.00830548574366]`.
- `D2_minus_D1_log_loss_improvement`: 90% `[-0.0599343133686, -0.0244020301673]`; 95% `[-0.0620289817331, -0.0195435239601]`.
- `D3_minus_D2_brier_improvement`: 90% `[-0.0139021729941, -0.00383053716582]`; 95% `[-0.0141494048943, -0.00282682408739]`.
- `confirmation_minus_pressure_return_after_20bps`: 90% `[-36.5765471443, 6.2739385849]`; 95% `[-38.0945928329, 10.2757647207]`.
- `pressure_minus_readiness_return_after_20bps`: 90% `[-59.0198011529, 12.7851081468]`; 95% `[-65.5622434477, 19.3058509274]`.

## Within-parent-slate bundled null

- `A2_minus_A1_brier_improvement`: real `0.00122939627995`, null q90 `-0.000374925407817`, percentile `0.980`.
- `D2_minus_D1_brier_improvement`: real `-0.0173504287875`, null q90 `-0.0123212914238`, percentile `0.600`.
- `pressure_minus_readiness_economic_30m`: real `-15.501912367`, null q90 `-13.5925695616`, percentile `0.840`.

## Delayed economic reference

- `readiness`: 0 / 10 / 20 bps = `40.425318` / `30.425318` / `20.425318` bps.
- `pressure`: 0 / 10 / 20 bps = `24.923405` / `14.923405` / `4.923405` bps.
- `confirmed`: 0 / 10 / 20 bps = `9.250316` / `-0.749684` / `-10.749684` bps.
- Singleton metric rows: `36`; multi-candidate metric rows: `36`.

## Concentration stress

- Maximum admitted-row stock share: `10.961538462%`.
- Largest stock: `QBTS`.
- Delete-largest same signed conclusions: `True`.
- Deleted principal increments non-negative: `False`.
- Economic result not dominated by largest stock: `True`.

The economic reference is synthetic and gross. It cannot rescue failed probability gates and does not model borrow, spread, or market impact.
