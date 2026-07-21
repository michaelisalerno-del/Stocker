# One-Minute Activity–Price Lead Screen V0

**Decision:** `no_one_minute_activity_increment`

This is a retrospective, research-only, observable-only bounded feasibility screen.
It is not prospective validation, achieved P&L, a deployable strategy, or evidence
of executable edge.

## Inputs and chronology

- Frozen predecessor: High-Movement Pressure-Onset Screen V0.1 at `cda387c`.
- Frozen admitted development rows: `1239`.
- Frozen admitted assessment rows: `1560`.
- Safe local one-minute timestamps read: `6180103`.
- Safe timestamp range: `2024-01-02T09:00:00+00:00` through
  `2025-08-22T23:59:00+00:00`.
- Timestamp convention: `bar_start`, proved by local 1m-to-5m
  OHLC alignment.
- Protected rows opened: `0`.
- Predictor window: ten fully completed bars, minute -10 through minute -1.
- Entry: open of minute +2; onset closes: +2 through +6; terminals: +16 and +31.

## Support and barriers

- Analysed development rows: `1230`.
- Analysed assessment rows / sessions / stocks / months:
  `1558` / `153` /
  `20` / `8`.
- Assessment UP / DOWN / NO_ONSET: `166` /
  `135` / `1257`.
- Onset barriers: checkpoint 30m `150.532903` bps; checkpoint 60m
  `125.572653` bps.

## Onset ladder

- A0: Brier `0.135685`, log loss `0.437492`, AUC `0.720367`.
- A1: Brier `0.139453`, log loss `0.446935`, AUC `0.697801`.
- A2: Brier `0.141639`, log loss `0.453671`, AUC `0.692314`.
- A3: Brier `0.145190`, log loss `0.468467`, AUC `0.683957`.

## Conditional-direction ladder

- D0: Brier `0.253880`, log loss `0.700930`, AUC `0.477064`.
- D1: Brier `0.280076`, log loss `0.758008`, AUC `0.432129`.
- D2: Brier `0.303273`, log loss `0.821062`, AUC `0.430968`.
- D3: Brier `0.311297`, log loss `0.845965`, AUC `0.432664`.

## Fixed increments

- A1_minus_A0: Brier `-0.00376856`, log loss `-0.00944322`, passes `False`.
- A2_minus_A1: Brier `-0.00218595`, log loss `-0.00673563`, passes `False`.
- A3_minus_A2: Brier `-0.0035508`, log loss `-0.0147962`, passes `False`.
- D1_minus_D0: Brier `-0.0261953`, log loss `-0.0570782`, passes `False`.
- D2_minus_D1: Brier `-0.0231968`, log loss `-0.0630541`, passes `False`.
- D3_minus_D2: Brier `-0.00802474`, log loss `-0.0249037`, passes `False`.

## Session-block bootstrap

- A1_minus_A0 / brier: 90% `[-0.00653601, -0.000579639]`, 95% `[-0.00706655, 0.000309603]`.
- A1_minus_A0 / log_loss: 90% `[-0.0165158, -0.000894887]`, 95% `[-0.0179939, -8.59118e-05]`.
- A2_minus_A1 / brier: 90% `[-0.00431492, -0.000303807]`, 95% `[-0.00485848, -8.9944e-05]`.
- A2_minus_A1 / log_loss: 90% `[-0.0126697, -0.00198419]`, 95% `[-0.0132594, -0.000941713]`.
- A3_minus_A2 / brier: 90% `[-0.00662386, -0.000957948]`, 95% `[-0.0071986, -0.000489937]`.
- A3_minus_A2 / log_loss: 90% `[-0.0252579, -0.0058324]`, 95% `[-0.0274778, -0.00463277]`.
- D1_minus_D0 / brier: 90% `[-0.0407415, -0.0136605]`, 95% `[-0.0422555, -0.0112277]`.
- D1_minus_D0 / log_loss: 90% `[-0.0878482, -0.029366]`, 95% `[-0.0937265, -0.0256072]`.
- D2_minus_D1 / brier: 90% `[-0.0404109, -0.00928617]`, 95% `[-0.0422188, -0.00398414]`.
- D2_minus_D1 / log_loss: 90% `[-0.103385, -0.0260408]`, 95% `[-0.112297, -0.0175151]`.
- D3_minus_D2 / brier: 90% `[-0.0170638, 0.00248227]`, 95% `[-0.020884, 0.00418399]`.
- D3_minus_D2 / log_loss: 90% `[-0.0533342, 0.0040161]`, 95% `[-0.058544, 0.0052494]`.
- activity_system_minus_price_system / economic_15m_after_20bps: 90% `[-22.7714, 24.0743]`, 95% `[-28.2668, 28.3738]`.
- interaction_system_minus_activity_system / economic_15m_after_20bps: 90% `[-11.8304, 41.0949]`, 95% `[-14.6309, 46.952]`.

## Within-slate activity null

- A2_minus_A1: real percentile `0.600`, null q90 `-0.000762924`.
- A3_minus_A2: real percentile `0.060`, null q90 `0.000120694`.
- D2_minus_D1: real percentile `0.100`, null q90 `0.00245106`.
- D3_minus_D2: real percentile `0.460`, null q90 `-0.0026125`.
- activity_system_minus_price_system: real percentile `0.460`, null q90 `15.3757`.

## Delayed economic-reference diagnostic

- activity_system, 15_minute_terminal: `-21.361` bps after 20 bps synthetic friction.
- activity_system, 30_minute_terminal: `-27.639` bps after 20 bps synthetic friction.
- interaction_system, 15_minute_terminal: `-9.970` bps after 20 bps synthetic friction.
- interaction_system, 30_minute_terminal: `-27.004` bps after 20 bps synthetic friction.
- price_system, 15_minute_terminal: `-22.830` bps after 20 bps synthetic friction.
- price_system, 30_minute_terminal: `-3.193` bps after 20 bps synthetic friction.

The economic values are synthetic-friction diagnostics, not achieved P&L. They
cannot rescue a failed probability gate.

## Concentration

- Maximum assessment row share: `0.1098`.
- All economic-selection concentration gates pass:
  `True`.
