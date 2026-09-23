# Market readiness deployment — 14 September 2026

Later configuration update: at 13:09:18 UTC, following the user’s request to enable the availability policy for all markets, US and ASX were also migrated to V10. See the final section below.

Deployed `53b226c0525d07166d68ab0f5bbfe22c4a28cba3` at 11:19:28 UTC, replacing
`c8a3753e46ec6329fe6e6573050c64458ad342f4`. Final checks completed at 11:20 UTC.
The user authorized deployment of all prepared fixes and the replacement LSE/Korea runs.

## Changes now active

- Near-sweep FX acquisition and bounded retries for pending cap filters.
- Safe cancellation/resumption of untouched pre-open acquisition and protection against
  late writes to sealed scanner observations.
- V10 candidate availability policy and exclusion diagnostics, preserving frozen V9 behavior.
- Existing required-risk-field form correction remains present.

LSE and Korea now use their prepared V10 PAPER runs, enabled with the original risk limits
and listing memberships. Their old V9 runs are archived/disabled with their original
specifications and session evidence retained. US retains its existing V9 run and enabled
state. ASX remains disabled. No LIVE run was configured or enabled.

| Market | Active run suffix | Result after deployment |
| --- | --- | --- |
| US | `cfb45447d78f` | Enabled, READY, untouched preparation for 13:30 UTC |
| LSE | `057e595a67cd` | Enabled V10; today's opening window already missed |
| Korea | `426a1231490c` | Enabled V10; today's opening window already missed |
| ASX | `c110ccfefe14` | Disabled |

The new LSE/Korea runs therefore report `CANDIDATE_SELECTION_WINDOW_MISSED` for today.
Their next eligible opening is the first opportunity to exercise the deployed policy.
No retrospective selection was constructed from later data, and successful trading or
complete next-session coverage is not implied by deployment.

## Deployment and verification

The release archive SHA-256 is
`53d871a7f5c1fc824f2022489dc5ed150361a269618a7782131b3c08760f5f80`.
All 429 archived files matched the installed release before and after cutover. Locked
server dependencies and the offline installation/startup/frozen-model/assets smoke passed
as the service user. The code is unchanged from the previously validated implementation:
1,233 Python tests passed, 14 skipped; lint, formatting, typing and three browser suites
passed. Only investigation documentation changed between that validation and this release.

The consistent, checksum-verified rollback bundle is
`/var/lib/stocker/backups/market-readiness-53b226c/state`; its parent also contains the
original configuration/enabled flags, service definition, pre/post system evidence,
Gateway PID, frozen-record hashes and the one-time pre-open recovery audit.

Before cutover, PAPER was connected/reconciled/ready with zero positions and open orders,
and no unfinished execution plan. All 105 US acquisition components were still PENDING,
with zero population, history requests or completed stages, well before the US open.
Enabled runs were paused through supported controls, then only Stocker was stopped.

The outgoing c8a3753 code still contains the pre-open cancellation defect. Its controlled
pause produced the exact unobserved US interruption previously diagnosed. While Stocker
was stopped, a guarded transaction restored only that untouched preparation, retaining
the complete interruption in `operational_recoveries` and a separate backup/audit. The new
code contains the permanent fix. No observed scanner or candidate evidence was rewritten.

Configuration and release symlinks were replaced atomically, then Stocker restarted.
PAPER reconnected to the same expected account, reconciled and became ready; US returned
to READY. The authenticated Gateway PID stayed unchanged throughout. Frozen original
LSE/Korea session, stage, population, component, pool and history-request hashes matched
before and after deployment. Database `quick_check` returned `ok`.

The dashboard reports the exact release, serves the new `20260914-available-candidates`
assets byte-for-byte, retains required-field validation, and still rejects unauthenticated
backend requests. Broker positions and open orders remained zero.

The earlier Gateway reauthentication had already cleared LSE permission warning 492;
that broker session was preserved. No subscription or broker authentication change was
needed for deployment.

## V10 enabled as the policy for all four configured markets

At 13:09:18 UTC the US V10 run `us_all-session_hard-paper-fcc1f1a23136` was verified enabled and READY. Its saved opening recipe uses `REJECT_UNAVAILABLE`, acquisition is pending/unsealed, and all 105 components are scheduled before the 13:30 UTC open. The original V9 run was archived/disabled with its history preserved.

ASX was migrated to `australia_asx-session_hard-paper-857159ad937e` with V10 while preserving its disabled state. LSE and Korea retained their enabled V10 configurations. All original risk limits and listing memberships matched exactly; no LIVE run was added.

This was a configuration-only migration on existing release 53b226c. Only Stocker restarted; the authenticated Gateway PID was unchanged. PAPER reconnected/reconciled with zero positions/open orders. The former US run had no observations before the restart, and the deployed pre-open pause fix handled its shutdown without manual database recovery.

The consistent SQLite backup, original configuration, named-universe snapshot, mapping and pre/post system evidence are in `/var/lib/stocker/backups/all-markets-v10-20260914`. Database backup quick_check passed. All four non-archived configured market runs now use `SESSION_HARD_CAUSAL_Q1_AVAILABLE_V10`.
