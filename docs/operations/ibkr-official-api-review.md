# Official IBKR API review for the Stocker prospective recorder

Checked: **2026-08-03**

Scope: the native TWS API used only for bounded market-data recording on the
dedicated Stocker server. This review does not authorise order submission,
account routing, automated authentication, or paper/live trading. It also does
not establish an options edge.

## Decision summary

- Treat IBKR support as an optional server integration. Replay and CI must not
  import it unconditionally.
- Use only the Python client distributed through the
  [official TWS API download and licence page](https://interactivebrokers.github.io/).
  IBKR's current documentation says packages from `pip`, NuGet, or other online
  repositories are not hosted, endorsed, or supported by IBKR; a registry
  `pip install ibapi` is therefore not an acceptable provenance path.
- On the check date, the official page listed Stable API 10.45 (2026-03-30)
  and Latest API 10.49 (2026-07-31). It listed Python in the Latest 10.49
  Windows and Mac/Unix packages. The Stable Mac/Unix package was listed as Java
  and Posix C++ only. Recheck this page at install time rather than assuming
  that "Stable" contains Python.
- The official Latest Mac/Unix archive checked for the current server slice is
  `twsapi_macunix.1049.01.zip` (`API_Version=10.49.01`,
  Python package `ibapi==10.49.1`) with SHA-256
  `f5d31e05f63be0d0fddc13ea8267c3a1625b0783baa17a44832e9151f8402b27`.
- Install `ibapi` into the Stocker server virtual environment from the extracted
  official archive's `source/pythonclient` directory with
  `uv pip install --python RELEASE/.venv/bin/python SOURCE`. Register immutable
  provenance that binds the official URL and archive hash to the exact installed
  Python source-tree hash. Do not vendor or redistribute the IBKR source in this
  repository or a Stocker bundle.
- If the official client is absent or its provenance/version cannot be
  established, keep replay operational and report either
  `blocked_official_ibkr_api_not_installed` or
  `blocked_unverified_official_ibkr_api`.
- Check the official Latest Mac/Unix release metadata weekly. Record whether a
  newer release exists, but never download or install broker code from that
  timer. Review, licence acceptance, archive verification, test, and promotion
  remain deliberate operator actions outside a trading session.
- Connect the recorder to TWS or IB Gateway over `127.0.0.1`. Require explicit
  host, port, client ID, and expected environment configuration. Never expose
  the IBKR socket on a public interface.
- Keep **Read-Only API enabled** as defence in depth and expose only a narrow
  market-data wrapper. The raw `EClient` and all order/account methods must stay
  outside the application service interface.
- Classify every quote by the actual `EWrapper.marketDataType` callback. Only
  type 1 (live) may enter Stocker's primary quoted-expectancy ledger. Frozen,
  delayed, and delayed-frozen observations may be retained as diagnostics only.

The principal first-party references are the
[current TWS API documentation](https://www.interactivebrokers.com/campus/ibkr-api-page/twsapi-doc/),
[TWS API reference](https://www.interactivebrokers.com/campus/ibkr-api-page/twsapi-ref/),
[market-data subscription guide](https://www.interactivebrokers.com/campus/ibkr-api-page/market-data-subscriptions/),
[TWS API settings guide](https://www.ibkrguides.com/traderworkstation/api.htm),
[paper-account guide](https://www.interactivebrokers.com/campus/trading-lessons/request-paper-trading-account/),
and [IBKR's TWS/third-party connection guide](https://www.interactivebrokers.com/campus/ibkr-api-page/third-party-connections/).

## Installation, process, and authentication boundary

The [TWS API requirements and installation sections](https://www.interactivebrokers.com/campus/ibkr-api-page/twsapi-doc/#download-the-tws-api)
require a running current TWS or IB Gateway and a current compatible API. They
currently state Python 3.11 as the minimum supported Python version and
recommend keeping the TWS/IB Gateway and API versions in sync. The official
download page currently recommends TWS or IB Gateway 1045 or newer for
comprehensive feature support.

TWS and IB Gateway are equivalent from an API application's perspective.
IB Gateway is lighter, but it is not a credential-free daemon. IBKR states that
the operator must authenticate in the TWS/IB Gateway GUI and that a headless
session without a GUI is not supported. TWS/IB Gateway can auto-restart daily,
but periodic weekly reauthentication still requires a manual login. IBKR also
states that 2FA is required and lists supported methods in its
[2FA section](https://www.interactivebrokers.com/campus/ibkr-api-page/twsapi-doc/#supported-two-factor-authentication-2fa).

Stocker implications:

- Stocker must never collect, persist, log, or automate the IBKR username,
  password, or 2FA response.
- Failure to authenticate TWS/IB Gateway is an operational condition, not
  something the recorder should bypass. Surface `blocked_ibkr_connection`.
- Keep IB Gateway/TWS lifecycle and authentication separate from the
  `stocker-recorder` service. The recorder may reconnect to an already
  authenticated local instance.
- Make the IBKR reader/callback loop owned by the recorder, outside HTTP
  handlers. The Python API starts a reader thread during connection and
  processes callbacks through `EClient.run()`; bounded application queues must
  isolate that callback path from database and web latency.

## Socket configuration, Read-Only API, and client IDs

The current [connectivity documentation](https://www.interactivebrokers.com/campus/ibkr-api-page/twsapi-doc/#connectivity)
shows these defaults:

| Application/session | Documented default |
| --- | ---: |
| TWS live | 7496 |
| TWS paper | 7497 |
| IB Gateway live | 4001 |
| IB Gateway paper | 4002 |

These are examples, not discovery rules. The effective value is the Socket Port
configured in TWS/IB Gateway. Stocker must require it explicitly and must not
infer live versus paper from a port number. The same-host connection should use
`localhost` or `127.0.0.1`; leave "Allow connections from localhost only"
enabled. A changed or mismatched port is a blocker, not permission to scan for
another socket.

IBKR's generic setup checklist tells users seeking the full API feature set to
disable Read-Only API. That generic instruction is deliberately **not**
appropriate for Stocker. IBKR's current
[installation/configuration lesson](https://www.interactivebrokers.com/campus/trading-lessons/installing-configuring-tws-for-the-api/)
states that Read-Only is enabled by default and blocks all API orders. The main
documentation also recognises that Read-Only mode rejects mutating operations
such as order binding and prevents API settings from being modified until it is
manually disabled. For this record-only application:

- enable ActiveX and Socket Clients;
- keep Read-Only API enabled;
- do not enable any "bypass order precaution" settings;
- keep the adapter's public protocol free of `placeOrder`, `cancelOrder`,
  `reqGlobalCancel`, `exerciseOptions`, positions, account, and execution
  methods; and
- treat Read-Only as an extra platform guard, not as a substitute for the
  no-order code boundary.

Each API connection has a client ID, and a TWS session supports up to 32
simultaneous API clients. IDs must be unique; error 326 means an ID is already
in use. Client ID 0 and the configured Master Client ID have special
order-observation/binding behaviour. A market-data-only recorder should use a
dedicated, explicitly configured **non-zero** client ID and should not be
configured as the Master client.

Do not send requests until the connection handshake is complete. IBKR notes
that `nextValidId` is commonly used as the completion signal and calls made
before it may be dropped. The name is order-oriented, but the callback is still
the documented connection-ready barrier.

## Connectivity and error state machine

IBKR sends errors, warnings, and informational notifications through
`EWrapper.error`; code severity must therefore be classified rather than
treating every callback alike. The authoritative meanings below come from the
[system and API message tables](https://www.interactivebrokers.com/campus/ibkr-api-page/twsapi-doc/#system-message-codes).

| Event/code | Official meaning | Required recorder behaviour |
| --- | --- | --- |
| `connectionClosed` | API socket closed | Mark disconnected, stop new requests, time out pending captures, audit, then apply bounded reconnect policy. |
| 502 | OS could not open the socket; commonly TWS/IBG is down, API is disabled, or the port differs | Report port/connection blocker. Do not probe ports or expose the socket remotely. |
| 326 | Client ID already in use | Refuse the competing connection and surface the configured client-ID conflict. |
| 1100 | TWS/IBG lost connectivity to IB; possible network issue, nightly reset, or competing session | Mark market data unavailable. Affected captures stay missing; never backfill them from a later quote. |
| 1101 | Connectivity restored, data lost | Reconcile local desired subscriptions and resubmit each still-required subscription exactly once. Do not duplicate subscriptions. |
| 1102 | Connectivity restored, data maintained | Keep/reconcile existing subscriptions; do **not** blindly resubmit them. |
| 1300 | Socket port changed and the API connection is being dropped | Fail closed. Require the configured port to be reviewed/updated by the operator before reconnecting. |
| 2103 / 2110 | Market-data farm or TWS-to-server connectivity problem | Mark affected data unhealthy and retain the event. |
| 2104 / 2106 / 2158 | Market-data, historical-data, or security-definition farm is OK | Informational health evidence, not an application error. |
| 100 | Message/request rate exceeded; repeated reject-mode violations can terminate the session | Stop admissions, retain the pacing error, and reject/wait within the configured local budget. |
| 101 | Maximum active ticker/market-data lines reached | Reject the capture explicitly as `market_data_budget_exhausted`; never silently omit contracts. |
| 102 | Duplicate ticker/request ID | Treat as an allocator/in-flight correlation defect and do not reuse the ID until the request is terminal. |
| 354, 10089, 10090, 10186 | Missing, partial, or API-ineligible market-data subscription | Preserve the permission error and mark requested fields/capture incomplete. Delayed data, if allowed, remains diagnostic. |
| 10197 | No market data because a competing live/paper session is using it | Mark the session unhealthy and surface the competing-session condition; do not mislabel missing data as a quiet market. |

Every transition, reconnect attempt, restored/lost-data distinction, request
timeout, and resubscription decision should be persisted as an IBKR connection
or data-health event.

## Permissions and actual market-data type

The [market-data subscription guide](https://www.interactivebrokers.com/campus/ibkr-api-page/market-data-subscriptions/)
states that Level 1 subscriptions are normally required for API top-of-book
data. OPRA supplies US options data, and option Greeks require subscriptions
for both the derivative and its underlying. Market data visible in TWS is not
guaranteed to be available through the API, so permission errors must be
recorded per contract and field.

`reqMarketDataType` selects the permitted fallback behaviour for subsequent
`reqMktData` requests, but TWS reports the actual type for each request through
`EWrapper.marketDataType`. In particular, requesting frozen or delayed data
does not guarantee that type: TWS can return live data when live permission is
available. Eligibility must use the callback, not the requested setting.

| Callback value | IBKR definition | Stocker use |
| ---: | --- | --- |
| 1 | Live streaming data; subscription required | Eligible for the primary quoted-expectancy ledger if all independent freshness/completeness gates pass. |
| 2 | Frozen, last data recorded at market close | Diagnostic only; never primary economic evidence. |
| 3 | Delayed, normally 15-20 minutes behind | Diagnostic only; never primary economic evidence. |
| 4 | Delayed-frozen | Diagnostic only; never primary economic evidence. |

Persist both the requested and actual type, plus any delayed tick identifiers.
Never infer a missing type or convert unavailable values to zero. IBKR's
[Unset Values section](https://www.interactivebrokers.com/campus/ibkr-api-page/twsapi-doc/#unset-values)
also documents maximum-type sentinels (for example, approximately
`1.7976931348623157E308` for an unset double). Normalize documented sentinels
and unavailable-price patterns to null before completeness or arithmetic
checks; never persist them as genuine observations.

`reqMktData` provides watchlist/top-of-book updates, not raw tick-by-tick data.
IBKR describes these as time-sampled aggregate snapshots (currently 100 ms for
US options), and states that real-time tick-by-tick data is not available for
options. The recorder must describe its option observations accordingly and
must not label them as a complete exchange event feed.

The API does not provide a provider timestamp for every bid and ask update.
The available-tick table provides a last-trade timestamp (tick 45) and a
delayed-last timestamp (tick 88), but no equivalent timestamp for every
top-of-book update. Persist the callback receive timestamp always and leave the
provider timestamp null when IBKR did not supply one.

## Market-data lines and pacing

The subscription guide says market-data lines are shared by the TWS watchlist
and all API connections for the username. The baseline is currently 100 lines,
but allocation can be higher and is account-dependent. A GUI watchlist or
another client can consume capacity that the recorder expected to use.

The current [pacing section](https://www.interactivebrokers.com/campus/ibkr-api-page/twsapi-doc/#pacing-limitations)
defines the maximum API request rate as the account's maximum market-data lines
divided by two per second (100 lines gives 50 requests/second). Do not hardcode
50. TWS can either reject excess traffic with code 100 or pace it internally;
internal pacing is unsuitable as the recorder's only protection because it can
turn a target-time capture into a late capture.

Implementation policy:

- configure a line budget, reserved line headroom, and a request-rate budget
  below the account maximum;
- account for continuous underlying streams, temporary option streams, pending
  requests, and requests awaiting cancellation;
- use a token-bucket or equivalent local limiter for all outbound requests;
- admit simultaneous signals only while the bounded capacity exists;
- otherwise use a bounded wait or record
  `blocked_market_data_budget_exhausted`;
- cancel temporary streams promptly on completion, timeout, or shutdown; and
- expose used, reserved, available, waiting, and rejected counts in health.

## Bounded option discovery and exact qualification

The [option-chain section](https://www.interactivebrokers.com/campus/ibkr-api-page/twsapi-doc/#option-chains)
explicitly discourages using an incomplete `reqContractDetails` request to
download a complete option chain. Ambiguous derivative requests are internally
paced and can produce a very large response.

Use this bounded sequence:

1. Qualify and retain the permanent underlying `conId`.
2. Call `reqSecDefOptParams` with the underlying symbol, security type, and
   `conId`. It returns callbacks grouped by exchange/trading class/multiplier,
   with available expiration strings and strike values.
3. Treat those values as discovery candidates only. IBKR warns that not every
   returned expiry/strike combination is a valid contract.
4. Apply Stocker's fixed DTE bucket, bounded strike count, and deterministic ATM
   tie rule locally. Do not expand after failure.
5. For each selected call/put, construct the most specific option `Contract`
   possible: symbol, `OPT`, exchange, currency, expiration, strike, right,
   multiplier, and trading class.
6. Qualify that exact contract with `reqContractDetails`. Accept exactly one
   result and persist `conId`, local symbol, multiplier, exchange, and trading
   class. Zero matches is `missing_contract`; multiple matches is ambiguous and
   must be rejected.
7. Cache successful metadata by session/underlying/expiry/strike/right/trading
   class, while preserving its source session and invalidating deliberately
   when contract definitions change.

This discovers a chain's coordinates but never streams the chain. It qualifies
and subscribes only the bounded contracts selected by the frozen recorder
rules.

## Quote requests, cancellation, and incomplete callbacks

`reqMktData` requires a unique request/ticker ID and returns price, size,
string, generic, and option-computation callbacks correlated by that ID.
`cancelMktData` terminates the stream.

A one-shot snapshot ends with `tickSnapshotEnd` after approximately 11 seconds,
may omit fields that did not arrive, and cannot include generic ticks. Since
option volume and open interest require generic ticks, the bounded recorder
should normally use a short-lived streaming request for the exact contract,
collect until its explicit completion policy or timeout, and always call
`cancelMktData`. A timeout or partial callback set remains incomplete.

Request generic tick 100 only when option call/put volume is required, and 101
only when call/put open interest is required. Keep missing fields null. The
[available tick table](https://www.interactivebrokers.com/campus/ibkr-api-page/twsapi-doc/#available-tick-types)
marks the old generic "Open Interest" tick 22 as deprecated; use the documented
call/put open-interest callbacks (ticks 27/28 via generic 101), not tick 22.

The callback loop must:

- allocate request IDs monotonically or otherwise guarantee uniqueness among
  active requests;
- correlate every callback/error/cancel/timeout with the request and contract;
- use bounded internal queues;
- retain partial data without declaring it complete;
- cancel on success, timeout, reconnect reconciliation, or shutdown; and
- never replace a missed target-time capture with a later quote.

## Option Greeks and implied volatility

IBKR's [option-Greeks documentation](https://www.interactivebrokers.com/campus/ibkr-api-page/twsapi-doc/#request-option-greeks)
states that option `reqMktData` requests automatically return computation
callbacks and that live Greeks require market-data permission for both the
option and underlying.

`EWrapper.tickOptionComputation` supplies implied volatility, delta, gamma,
vega, theta, option price, present value of dividends, and underlying price.
The callback's field identifies materially different computation sources:

| Tick | Computation source |
| ---: | --- |
| 10 | Based on the option bid |
| 11 | Based on the option ask |
| 12 | Based on the option last trade |
| 13 | TWS model option computation |
| 80-83 | Delayed bid, ask, last, and model equivalents |

Persist these as separate observations with the tick/source field. Do not
collapse bid-, ask-, last-, and model-derived values into one unexplained
"Greek" or "IV" value. These are TWS computations, not executable prices.
Shadow entry/exit arithmetic must continue to use the independently observed
option ask/bid ticks; it must never use `optPrice`, a model value, midpoint, or
last as a fill.

The API also exposes `calculateOptionPrice` and
`calculateImpliedVolatility` for hypothetical inputs. Those functions are not
part of the observed-quote evidence path and should not be called by this
recorder.

## Paper/live session considerations

IBKR's [paper-account guide](https://www.interactivebrokers.com/campus/trading-lessons/request-paper-trading-account/)
states that a paper account has separate credentials. Live market-data
subscriptions can be shared to one paper user, but shared market data cannot be
used by the linked live and paper users simultaneously. The TWS connection API
itself does not automatically make a safe application-level distinction
between live and paper; multiple TWS instances must use different socket ports.

Consequences:

- require and audit an explicit `expected_environment`;
- do not infer it from a default port;
- do not start a competing live session using shared market data;
- handle code 10197 as a hard data-health condition;
- permit only `record_only` or `shadow` recorder modes regardless of account;
  and
- remember that "paper account" does not make an order-capable code path
  acceptable. Stocker has no such path in this phase.

## Acceptance implications for Stocker

The IBKR integration is safe enough for this phase only when all of the
following are true:

- official-client provenance and compatible versions are recorded;
- the import is optional and replay works with no IBKR installation;
- TWS/IB Gateway is manually authenticated, Read-Only, and loopback-only;
- host, port, unique non-zero client ID, and expected environment are explicit;
- the adapter exposes market-data and contract-discovery methods only;
- actual market-data type is recorded per request;
- non-live data cannot enter the primary expectancy ledger;
- market-data lines, reserved headroom, and request rate are locally enforced;
- option discovery is coordinate-only and exact contract qualification is
  bounded;
- temporary streams are timed out and cancelled;
- 1100/1101/1102/1300 and competing-session behaviour is implemented and
  tested; and
- missing/partial/permission-denied values remain missing with an explicit
  rejection reason.

If any of these cannot be established, the correct behaviour is a visible
blocker while replay, persistence, and read-only web monitoring remain
available.
