# One-Minute Activity–Price Lead Screen V0

**Decision:** `blocked_one_minute_history_unavailable`

This retrospective, research-only, observable-only feasibility screen stopped at the
first one-minute data gate. It is not prospective validation, achieved P&L, a
strategy, or evidence of executable edge.

## Availability gate

- Frozen cohort: `20` stocks.
- Required XNYS sessions: `412`.
- Symbol-session audit rows: `8240`.
- Complete symbol-sessions: `0`.
- Missing symbol-sessions: `8240`.
- Local one-minute files present: `0`.
- Safe one-minute timestamp rows materialised: `0`.
- Minimum safe one-minute timestamp: `None`.
- Maximum safe one-minute timestamp: `None`.
- Protected rows opened: `0`.
- External data downloaded: `False`.
- External API called: `False`.
- Credentials read: `False`.

Every session is reported in `one_minute_availability_audit.csv` with its symbol,
month, XNYS open/close, separate bar-start and bar-end candidate ordinals, exact
missing ordinals, duplicates, off-grid rows, source identity, and QA status. Neither
candidate is promoted to a timestamp convention without empirical proof.

## Frozen nomination population

- Source: High-Movement Pressure-Onset Screen V0.1 at `cda387c`.
- Frozen development admitted rows: `1239`.
- Frozen assessment rows / sessions / stocks:
  `1560` / `153` / `20`.
- Assessment checkpoint rows: `{'6': 766, '12': 794}`.
- Admission rule recomputed: `False`.
- Exact frozen identity reconstruction: `True`.

## Downstream work not opened

Timestamp semantics could not be empirically proved because complete one-minute
history did not pass the availability gate. No one-minute normalisation,
price/activity feature, interaction,
onset barrier, label, model, bootstrap, activity null, permutation importance,
economic reference, concentration selection, or plot was produced. Zero models were
fitted. This is the required fail-closed behavior; five-minute volume was not used as
a substitute.
