# Corrected Opening Transition Event Summary V2

This versioned correction preserves the original V1 artifacts.

The previous event-accounting labels mixed two populations:

- `unique_opening_transition_event_count` counted severe VTI events with at
  least one eligible checkpoint-6 stock episode.
- the old negative and positive columns counted every severe VTI session,
  including sessions with no eligible stock episode.

On one common eligible-event population, the corrected counts are:

| Period | Total eligible events | Negative | Positive | All-market negative | All-market positive |
| --- | ---: | ---: | ---: | ---: | ---: |
| development | 26 | 18 | 8 | 23 | 15 |
| assessment | 43 | 23 | 20 | 28 | 30 |
| stress | 13 | 5 | 8 | 7 | 11 |

Each event has exactly one sign, and negative plus positive equals the signed
eligible-event total in every period. The reconciliation outcome is
`event_label_ambiguity_corrected`; no event-construction or aggregation bug was
found. The prior scientific interpretation remains
`blocked_insufficient_support`.
