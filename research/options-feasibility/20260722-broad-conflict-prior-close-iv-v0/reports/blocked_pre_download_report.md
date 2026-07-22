# Prior-Close Options IV Movement Screen V0

Primary decision: `blocked_historical_options_date_unavailable`

Provider bulk page requests recorded: 0. Provider setup requests recorded: 2. The corrected preflight requested 2025-08-21 but the returned historical EOD resources and quote timestamps covered 2025-09-03 through 2025-09-16; `tradetime` reflected last-trade activity rather than the EOD observation date. The official schema exposes no observation-date filter. The fail-closed download did not produce an options-movement inference. No intraday option fill or option P&L was calculated.
