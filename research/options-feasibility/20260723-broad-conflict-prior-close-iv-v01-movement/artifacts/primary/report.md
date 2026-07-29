# Three-stock prior-close IV movement outcomes V0.1

Decision: `descriptive_options_movement_structure_only`

The fixed sample produced 96 clean underlying-movement rows; 86 rows
had a valid exact-previous-session ATM option pair and therefore support IV-relative outcomes.
Assessment support is 39 rows from one session and three stocks.

Assessment weighted 15-minute absolute movement was
0.00764885; expected absolute movement from
previous-close ATM IV was 0.00624880; and the mean
IV residual was 0.00140005. The exceed-IV rate was
34.0741%.

Assessment weighted absolute movement was
0.00629154 at 10 minutes,
0.00957240 at 30 minutes
(37 rows), and
0.01430936 at 60 minutes
(29 rows). Registered completion occurred
in bars two or three for 0 assessment
rows.

`BROAD_CONFLICT` has 8 assessment rows and mean IV residual
0.00156141. `LOW_ROUTE_SUPPORT` has
6 assessment rows and mean IV residual
-0.00077095. Their descriptive residual difference is
0.00233235; their IV-sigma-ratio difference is
0.28067053; and their exceed-IV-rate difference is
-13.3333%.

This three-stock, three-date sample cannot pass the frozen coverage, stability, bootstrap,
matched-control, or model gates. No O0/O1/R0/R1 model was fit. The result does not establish the
binding broad-conflict hypothesis and makes no claim about option P&L, executable fills,
profitability, economic edge, prospective validation, or trading utility.
