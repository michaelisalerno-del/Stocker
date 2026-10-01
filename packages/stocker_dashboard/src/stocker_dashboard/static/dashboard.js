"use strict";
// SLRNO dashboard: a read-only consumer of runtime state. Every write uses textContent or
// attributes (never innerHTML with server data) and keyed nodes, so refreshes keep focus,
// open panels and scroll positions.
const $ = (id) => document.getElementById(id);
const markets = ["CL", "GC", "NG", "NQ", "SI"];
const names = {CL: "Crude oil", GC: "Gold", NG: "Natural gas", NQ: "Nasdaq 100", SI: "Silver"};
const SVG = "http://www.w3.org/2000/svg";
const route = location.pathname.slice(1) || "overview";
// The Opportunities page renders the "trades" section; Markets reuses the overview cards.
const page = route === "opportunities" ? "trades" : route === "markets" ? "overview" : route;
let selectedMarket = sessionStorage.getItem("slrno-market") || "CL";
if (!markets.includes(selectedMarket)) selectedMarket = "CL";
let offset = 0, paused = false, pending = null, selectedIdentity = "";
let detailRequest = null, detailSignature = null, controlPending = false, lastRefresh = null, accountSnapshot = null;
let pageSize = 100; // replaced by the server's page_size
let limits = {};
let serverOffset = 0, nextClock = null; // countdowns use server time, not the browser clock
const serverNow = () => Date.now() + serverOffset;
const chartGeometry = new Map(), depthExpiry = new Map(), quoteReceipts = new Map(), flowExpiry = new Map();

// Formatters are built once: the pages format hundreds of values per refresh.
const GBP = new Intl.NumberFormat("en-GB", {style: "currency", currency: "GBP"});
const NATIVE = new Intl.NumberFormat("en-GB", {maximumFractionDigits: 2, minimumFractionDigits: 2});
const LONDON = new Intl.DateTimeFormat("en-GB", {timeZone: "Europe/London", day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit", second: "2-digit"});
const NEW_YORK = new Intl.DateTimeFormat("en-GB", {timeZone: "America/New_York", hour: "2-digit", minute: "2-digit"});
const money = (v) => v == null ? "Unavailable" : GBP.format(v);
const time = (v) => v ? LONDON.format(new Date(v)) : "—";
const hhmm = (v) => v ? NEW_YORK.format(new Date(v)) : "—";
const numeric = (v, suffix = "") => typeof v === "number" && Number.isFinite(v) ? `${Number(v.toFixed(4))}${suffix}` : "UNAVAILABLE";
const text = (id, v) => {
  const el = $(id), next = String(v ?? "—");
  if (el && el.textContent !== next) el.textContent = next;
};
const show = (id, visible) => { const el = $(id); if (el && el.hidden === visible) el.hidden = !visible; };
const explanations = {
  REFERENCE_CONTRACT_SELECTION_REQUIRED: "Select and verify the Saxo futures contract",
  LISTED_PRODUCT_AND_DELTA_TOLERANCE_UNAPPROVED: "Option product and selection approval required",
  MINIMUM_CONTRACT_COST_EXCEEDS_BUDGET: "One contract, including entry and reserved exit costs, exceeds the per-trade ceiling",
  WITHIN_BUDGET: "One contract fits the per-trade ceiling",
  INCOMPLETE_COMPLETED_HISTORY: "Waiting for the final completed-minute bar",
  STALE_SIGNAL_NO_REPLAY: "The entry deadline passed before this opportunity was ready",
  EXECUTION_DISABLED: "Paper execution is disabled", PAPER_DISARMED: "Paper entries are not armed",
  SKIP_CAPACITY_FULL: "All four trade slots are occupied or reserved",
  RECONCILIATION_REQUIRED: "Broker reconciliation is required", RECONCILIATION_STALE: "Broker reconciliation is overdue",
  ENTRIES_PAUSED: "New entries are paused",
  GC_LISTED_EXECUTION_RULE_UNAPPROVED: "Gold is monitor-only; execution is not approved",
  ORDER_STATUS_UNCERTAIN: "Broker order status needs reconciliation",
  RESERVED: "Reserved · entry order pending",
  EXPOSURE_REQUIRES_RECONCILIATION: "Exposure awaiting reconciliation",
  OPEN: "Open · exposure reconciled with the broker",
  AUTHENTICATION_REQUIRED: "Connect the configured Saxo account",
  NO_CANDIDATE: "No option strike ranked yet", AWAITING_CONTRACT: "Needs a verified futures contract first",
  STALE_OR_MISSING: "No current quote",
  QUOTE_DELAYED_OR_DELAY_UNKNOWN: "Quote is delayed (the strategy needs real-time data)",
  QUOTE_NOT_USABLE: "Quote price type is not usable (old indicative, pending or no market)",
  QUOTE_STALE_OR_UNAVAILABLE: "No quote received in the last 5 seconds",
  SAXO_CHART_DATA_DELAYED: "Saxo chart data is delayed; completed bars cannot arrive in time",
};
const display = (v) => explanations[v] || String(v || "").replaceAll("_", " ");

for (const name of ["overview", "trades", "execution", "system"]) $(name).hidden = name !== page;
text("title", {markets: "Markets", opportunities: "Opportunities", execution: "Execution", system: "System"}[route] || "Futures overview");
document.querySelectorAll("nav a").forEach((a) => { if (a.pathname === location.pathname) a.setAttribute("aria-current", "page"); });
show("setup", route === "overview" || route === "system");
show("strategy-strip", route === "overview");
show("account-strip", route === "overview");
show("market-selector", route === "markets");

// Stable card nodes. Templates below are constant strings; server data is written later
// with text() or attributes only.
function overviewCard(m) {
  return `<div class="card-head"><h2>${m} <span>${names[m]}</span></h2><strong class="state" id="state-${m}"></strong></div>
    <p class="contract" id="contract-${m}"></p>
    <ol class="gates" id="gates-${m}" aria-label="${m} readiness at last refresh"></ol>
    <p class="block" id="block-${m}"></p>
    <p class="position" id="position-${m}" hidden></p>
    <p class="candidate" id="candidate-${m}" hidden></p>
    <p class="events" id="events-${m}" hidden></p>
    <small class="data-line" id="data-${m}"></small>
    <a class="detail-link" href="/markets?market=${m}">Market detail →</a>`;
}
function marketCard(m) {
  return `<div class="card-head"><div><h2>${m} <span>${names[m]}</span></h2><p class="contract" id="contract-${m}"></p></div><span class="state" id="state-${m}"></span></div>
    <div class="status-line"><span id="market-${m}"></span><span id="l1-${m}"></span><span id="l2-${m}"></span><span id="data-${m}"></span></div>
    <ol class="gates" id="gates-${m}" aria-label="${m} readiness at last refresh"></ol>
    <p class="block" id="block-${m}"></p>
    <svg class="chart" viewBox="0 0 640 210" role="img" aria-label="${m} completed underlying prices with frozen clock marks">
      <g id="axis-${m}" class="axis"></g><rect id="rv-${m}" class="rv-window" x="0" y="10" width="0" height="170"/>
      <g id="clocks-${m}" class="clock-marks"></g><g id="band-${m}" class="trade-band"></g>
      <polyline id="line-${m}" fill="none"/><g id="markers-${m}"></g>
      <text id="chart-empty-${m}" x="320" y="100" text-anchor="middle">Awaiting completed bars</text>
    </svg>
    <p class="chart-key">Grey marks: frozen clocks (NY) · shaded: 15-minute RV window · green band: open trade to exit</p>
    <div class="market-grid">
      <section class="panel ticket" aria-label="${m} option trade ticket"><h3>Trade ticket</h3>
        <p class="ticket-contract" id="ticket-contract-${m}"></p>
        <dl>
          <dt>Quote</dt><dd id="ticket-quote-${m}"></dd>
          <dt>Spread</dt><dd id="ticket-spread-${m}"></dd>
          <dt>Delta</dt><dd id="ticket-delta-${m}"></dd>
          <dt>Volatility</dt><dd id="ticket-iv-${m}"></dd>
          <dt>Forecast</dt><dd id="ticket-forecast-${m}"></dd>
          <dt>Cutoff</dt><dd id="ticket-cutoff-${m}"></dd>
        </dl>
        <div class="cost-meter" aria-label="All-in cost against the per-trade ceiling"><div class="cost-fill" id="ticket-fill-${m}"></div></div>
        <p class="cost-line" id="ticket-cost-${m}"></p>
        <p class="verdict" id="ticket-verdict-${m}"></p>
        <details><summary>Option evidence</summary>
          <p id="option-identity-${m}"></p><p id="option-deadline-${m}"></p><p id="option-quote-${m}"></p>
          <p id="option-analytics-${m}"></p><p id="option-volume-${m}"></p><p id="option-underlying-${m}"></p>
          <p id="option-cost-${m}"></p><p id="option-coverage-${m}"></p>
          <p class="muted">Provider analytics are unverified context. Chain indications and latest trades are not executable prices.</p>
          <pre id="option-meta-${m}"></pre></details>
      </section>
      <section class="panel context" aria-label="${m} market context"><h3>Market context</h3>
        <dl>
          <dt>Direction</dt><dd id="direction-${m}"></dd>
          <dt>Conditions</dt><dd id="conditions-${m}"></dd>
          <dt>Quote</dt><dd id="quote-${m}"></dd>
          <dt>Price</dt><dd id="price-context-${m}">Not reported</dd>
          <dt>Sessions</dt><dd id="sessions-${m}"></dd>
          <dt>Events</dt><dd id="events-today-${m}"></dd>
          <dt>Vol smile</dt><dd><svg viewBox="0 0 300 90" class="smile" id="smile-${m}" aria-label="${m} provider implied volatility by strike"></svg><span class="muted" id="smile-note-${m}"></span></dd>
          <dt>Recorder</dt><dd class="recorder-line" id="recorder-${m}"></dd>
        </dl>
        <div class="position" id="position-${m}"></div>
      </section>
    </div>
    <section class="book-flow" id="book-flow-${m}"><h3>Book · sampled L2</h3>
      <p class="muted">Sampled order-book observations; not a complete execution tape. Observation only — never an entry rule.</p>
      <div class="imbalance" aria-label="Five-level size imbalance"><div class="imbalance-scale"><span>Ask-heavy</span><span>Balanced</span><span>Bid-heavy</span></div>
        <div class="imbalance-track"><div class="imbalance-mark" id="imbalance-mark-${m}"></div></div><p id="imbalance-${m}"></p></div>
      <div class="ladder" id="ladder-${m}" aria-label="${m} displayed depth">
        <div class="ladder-head"><span>Ask size</span><span></span><span>Ask</span><span>Bid</span><span></span><span>Bid size</span></div>
        ${Array.from({length: 10}, (_, i) => `<div class="ladder-row" id="depth-${m}-${i}" hidden><span class="size"></span><span class="bar ask"><i></i></span><span class="price ask"></span><span class="price bid"></span><span class="bar bid"><i></i></span><span class="size"></span></div>`).join("")}
      </div>
      <p class="muted" id="depth-note-${m}"></p>
      <dl class="flow-facts">
        <dt>Status</dt><dd id="flow-status-${m}"></dd><dt>Feed</dt><dd id="flow-feed-${m}"></dd>
        <dt>Spread</dt><dd id="flow-spread-${m}"></dd><dt>Midpoint</dt><dd id="flow-mid-${m}"></dd>
        <dt>Changes</dt><dd id="flow-changes-${m}"></dd><dt>Persistence</dt><dd id="flow-persistence-${m}"></dd>
        <dt>Last trade</dt><dd id="flow-last-${m}"></dd><dt>Volume</dt><dd id="flow-volume-${m}"></dd>
      </dl>
      <details id="book-history-${m}"><summary>15-minute book history</summary>
        <div class="sparks">
          <figure><figcaption>Spread (ticks)</figcaption><svg viewBox="0 0 300 60" id="spark-spread-${m}" class="spark"></svg></figure>
          <figure><figcaption>Top-of-book size · bid / ask</figcaption><svg viewBox="0 0 300 60" id="spark-size-${m}" class="spark"></svg></figure>
          <figure><figcaption>Five-level imbalance</figcaption><svg viewBox="0 0 300 60" id="spark-imbalance-${m}" class="spark"></svg></figure>
        </div>
        <canvas class="heatmap" id="heatmap-${m}" width="900" height="260" aria-label="${m} sampled depth heatmap"></canvas>
        <p class="muted" id="heatmap-note-${m}">Open to load the retained window.</p>
      </details>
      <details><summary>Depth totals by level</summary>
        <div class="flow-table"><table aria-label="${m} book-flow depth totals"><thead><tr><th>Levels</th><th>Bid depth</th><th>Ask depth</th><th>Size imbalance</th><th>Observed state</th></tr></thead>
        <tbody>${[1, 3, 5, 10].map((n) => `<tr id="flow-depth-${m}-${n}"><th>${n}</th><td></td><td></td><td></td><td></td></tr>`).join("")}</tbody></table></div>
        <p class="muted">No order-count imbalance: Saxo sends no order counts on this feed (UsingOrders false; its order fields repeat the sizes).</p>
      </details>
      <details id="flow-detail-${m}"><summary>Coverage, fields and observation times</summary><pre id="flow-meta-${m}"></pre></details>
    </section>
    <details id="raw-state-${m}"><summary>Contract, rule &amp; raw state</summary><pre id="details-${m}"></pre></details>`;
}
for (const m of markets) {
  const card = document.createElement("article");
  card.className = "market";
  card.id = `card-${m}`;
  card.innerHTML = route === "markets" ? marketCard(m) : overviewCard(m);
  $("markets").append(card);
}

function renderGates(m) {
  const list = $(`gates-${m.market}`);
  if (!list) return;
  const items = m.gates || [];
  items.forEach((g, i) => {
    let li = list.children[i];
    if (!li) { li = document.createElement("li"); list.append(li); }
    const cls = g.ok ? "ok" : "fail";
    if (li.className !== cls) li.className = cls;
    if (li.textContent !== g.label) li.textContent = g.label;
    const title = g.ok ? `${g.label}: ready` : `${g.label}: ${display(g.detail)}`;
    if (li.title !== title) li.title = title;
  });
  while (list.children.length > items.length) list.lastChild.remove();
}
function firstBlock(m) {
  const failing = (m.gates || []).find((g) => !g.ok);
  if (m.pending_data) return "Waiting for the completed boundary bar · original entry deadline applies";
  if (failing) return `${failing.label}: ${display(failing.detail)}`;
  return display(m.block_reason) || (m.entry_enabled ? "Ready · the clock decision re-checks every gate" : "");
}

function optionChoice(m) {
  const context = m.option_context || {}, contracts = context.contracts || [];
  const held = m.trades?.[0], event = context.latest_event;
  const eventUic = event?.context?.identity?.uic;
  const uic = held?.option?.uic || (contracts.some((c) => c.identity?.uic === eventUic) ? eventUic : context.candidate_uic);
  return {context, held, event, eventUic, uic, o: contracts.find((c) => c.identity?.uic === uic)};
}
function optionContext(m) {
  const {context, held, event, eventUic, uic, o} = optionChoice(m);
  const id = o?.identity || {}, q = o?.quote || {}, cost = o?.costs || {};
  const obs = (name, rows) => {
    const r = rows?.[name];
    return r ? `${numeric(r.value)} · ${r.status} · age ${Math.floor(r.age_seconds)}s · effective ${r.effective_at || "unknown"}` : "MISSING";
  };
  const analytics = {...(o?.chain_analytics || {}), ...(o?.analytics || {})};
  text(`option-identity-${m.market}`, o ? `${held ? "Owned contract" : uic === eventUic ? "Event-selected contract" : "Current candidate"}: ${id.symbol || id.uic} · ${id.right} ${id.strike} · UIC ${id.uic} · underlying ${id.underlying_symbol || "unknown"} / ${id.underlying_uic}` : `No selected contract · ${context.problem || "UNAVAILABLE"}`);
  text(`option-deadline-${m.market}`, `Expiry ${id.expiry || "unknown"} · last trading ${id.last_trade_at || "UNVERIFIED"} · strategy exit ${held?.exit_at || (uic === eventUic && event?.context?.strategy_exit_at) || "set by event + 60 minutes"}`);
  text(`option-quote-${m.market}`, `Regular quote ${o?.quote_status || "MISSING"} · bid ${numeric(q.Bid)} × ${numeric(o?.sizes?.Bid)} (${q.PriceTypeBid || "unknown"}) · ask ${numeric(q.Ask)} × ${numeric(o?.sizes?.Ask)} (${q.PriceTypeAsk || "unknown"}) · spread ${numeric(typeof q.Ask === "number" && typeof q.Bid === "number" ? q.Ask - q.Bid : null)} · delay ${q.DelayedByMinutes ?? "unknown"} min · bid size ${o?.size_status?.Bid || "UNVERIFIED"} · ask size ${o?.size_status?.Ask || "UNVERIFIED"}`);
  text(`option-analytics-${m.market}`, `Provider delta ${obs("Greeks.Delta", analytics)} · IV ${obs(analytics["Greeks.MidVolatility"] ? "Greeks.MidVolatility" : "Greeks.MidVol", analytics)} · provider units / scaling unverified`);
  text(`option-volume-${m.market}`, `Option volume ${obs("PriceInfoDetails.Volume", analytics)} · OI ${obs("InstrumentPriceDetails.OpenInterest", analytics)}`);
  text(`option-underlying-${m.market}`, `Future volume ${obs("PriceInfoDetails.Volume", m.underlying_context)} · OI ${obs("InstrumentPriceDetails.OpenInterest", m.underlying_context)}`);
  text(`option-cost-${m.market}`, `Minimum purchase ${numeric(cost.minimum_purchase_cost_gbp, " GBP")} · entry fees ${numeric(cost.entry_costs_gbp)} · estimated exit ${numeric(cost.estimated_exit_costs_gbp)} · budget total ${numeric(cost.total_gbp)} / ${money(limits.per_trade_gbp)} · remaining ${numeric(cost.remaining_budget_gbp)} · ${cost.reason || cost.budget_result || "UNVERIFIED"}`);
  text(`option-coverage-${m.market}`, `Actual current option buffer ${Math.floor(o?.coverage_seconds || 0)} / 900s · subscription started ${o?.subscription_started_at ? new Date(o.subscription_started_at * 1000).toISOString() : "unknown"} · latest event ${event?.id || "none"}: UIC ${event?.context?.identity?.uic || "unavailable"}, actual pre-trigger ${event ? Math.floor(event.context.pre_trigger_seconds || 0) + " / 900s" : "unavailable"}`);
  text(`option-meta-${m.market}`, $(`option-meta-${m.market}`).closest("details").open ? JSON.stringify(context, null, 2) : "");
  ticket(m, {held, uic, o, context});
}
function ticket(m, {held, uic, o, context}) {
  const id = o?.identity || {}, q = o?.quote || {}, cost = o?.costs || {}, sizes = o?.sizes || {};
  text(`ticket-contract-${m.market}`, o ? `${held ? "Owned" : "Candidate"} · ${id.symbol || "UIC " + id.uic} · ${id.right} ${id.strike} · expiry ${id.expiry || "unknown"}` : `No candidate strike · ${display(context.problem) || "not ranked"}`);
  const bid = q.Bid, ask = q.Ask, quoted = typeof bid === "number" && typeof ask === "number";
  text(`ticket-quote-${m.market}`, o ? `Bid ${numeric(bid)} × ${sizes.Bid ?? "—"} · Ask ${numeric(ask)} × ${sizes.Ask ?? "—"} · ${o.quote_status || "MISSING"}` : "—");
  const mid = quoted ? (bid + ask) / 2 : null;
  text(`ticket-spread-${m.market}`, quoted && mid > 0 ? `${numeric(ask - bid)} · ${(100 * (ask - bid) / mid).toFixed(1)}% of mid` : "—");
  const model = m.candidate_deltas?.[String(uic)], provider = o?.analytics?.["Greeks.Delta"]?.value;
  text(`ticket-delta-${m.market}`, o ? `Frozen model |Δ| ${numeric(model)} vs target ${numeric(m.target_delta)} · provider ${numeric(provider)} (unverified)` : "—");
  const iv = o?.analytics?.["Greeks.MidVol"]?.value, sigma = m.model_sigma;
  text(`ticket-iv-${m.market}`, typeof iv === "number" ? `Implied ${(100 * iv).toFixed(1)}% vs frozen model σ ${typeof sigma === "number" ? (100 * sigma).toFixed(1) + "%" : "unavailable"}${typeof sigma === "number" ? ` · ${iv - sigma >= 0 ? "+" : ""}${(100 * (iv - sigma)).toFixed(1)} pts` : ""}` : "Implied volatility not reported");
  // look14, observation only: the option's price against the movement the forecast expects.
  const f = m.forecast, fo = f?.option?.uic === uic ? f.option : null;
  text(`ticket-forecast-${m.market}`, f?.status === "OBSERVED" ? `Movement ${f.level.toFixed(2)}× normal for the time of day · next hour ±${(100 * f.next_hour_move).toFixed(2)}%${typeof fo?.implied_over_forecast === "number" ? ` · priced for ${fo.implied_over_forecast.toFixed(2)}× the forecast move to expiry (${(100 * fo.implied_move_to_expiry).toFixed(2)}% vs ${(100 * fo.forecast_move_to_expiry).toFixed(2)}%)` : ""}` : `Forecast unavailable · ${display(f?.reason) || "no bars"}`);
  text(`ticket-cutoff-${m.market}`, o ? `Last trading ${id.last_trade_at ? hhmm(id.last_trade_at) + " NY" : "UNVERIFIED"} · exit ${held?.exit_at ? hhmm(held.exit_at) + " NY" : "clock + 60 min"}` : "—");
  const ceiling = limits.per_trade_gbp, total = cost.total_gbp;
  const fill = $(`ticket-fill-${m.market}`);
  if (fill) {
    const width = `${Math.min(100, Math.max(0, 100 * (total || 0) / ceiling)).toFixed(1)}%`;
    if (fill.style.width !== width) fill.style.width = width;
    const over = typeof total === "number" && total > ceiling;
    if (fill.classList.contains("over") !== over) fill.classList.toggle("over", over);
  }
  text(`ticket-cost-${m.market}`, typeof total === "number" ? `${money(total)} all-in of ${money(ceiling)} · premium ${numeric(cost.premium_gbp)} · entry ${numeric(cost.entry_costs_gbp)} · exit reserve ${numeric(cost.estimated_exit_costs_gbp)}` : "All-in cost unavailable");
  text(`ticket-verdict-${m.market}`, display(cost.reason || cost.budget_result) || "Unverified");
}

function bookFlow(m, f, rec = {}, identity = {}) {
  if (route !== "markets") return;
  flowExpiry.set(m, (f.valid_until || 0) * 1000);
  text(`flow-status-${m}`, `${f.status || "UNAVAILABLE"}${(f.quality_flags || []).length ? " · " + f.quality_flags.join(" · ") : ""} · ${rec.state || "UNAVAILABLE"} · ${Math.floor(rec.prehistory_seconds || 0)} / 900s buffer`);
  const feed = f.feed || {}, cadence = feed.observed_receipt_ms || {};
  text(`flow-feed-${m}`, `${identity.environment || "UNVERIFIED"} · UIC ${identity.uic ?? "UNVERIFIED"} · delay ${numeric(f.delay_minutes, " min")} · granted ${numeric(feed.granted_refresh_ms, " ms")} · observed receipt mean ${numeric(cadence.mean, " ms")} (${cadence.samples || 0} intervals)`);
  text(`flow-spread-${m}`, `${numeric(f.spread_ticks, " ticks")} · usable depth ${f.available_levels?.bid ?? 0} bid / ${f.available_levels?.ask ?? 0} ask levels`);
  for (const n of [1, 3, 5, 10]) {
    const d = f.depth?.[n] || {}, row = $(`flow-depth-${m}-${n}`);
    [numeric(d.bid), numeric(d.ask), numeric(d.imbalance), d.label || "UNAVAILABLE"].forEach((v, i) => { if (row.children[i + 1].textContent !== v) row.children[i + 1].textContent = v; });
  }
  const imbalance = f.depth?.[5]?.imbalance;
  const mark = $(`imbalance-mark-${m}`);
  if (mark) {
    const left = typeof imbalance === "number" ? `${(50 + 50 * imbalance).toFixed(1)}%` : "50%";
    if (mark.style.left !== left) mark.style.left = left;
    mark.classList.toggle("unavailable", typeof imbalance !== "number");
  }
  text(`imbalance-${m}`, typeof imbalance === "number" ? `${imbalance > 0 ? "+" : ""}${imbalance.toFixed(2)} · ${f.depth[5].label.replaceAll("_", " ").toLowerCase()} over five levels` : "Unavailable — needs five levels each side");
  text(`flow-mid-${m}`, `Size-weighted ${numeric(f.weighted_midpoint)} · displacement ${numeric(f.weighted_displacement_ticks, " ticks")}`);
  text(`flow-changes-${m}`, "Five-level depth · " + [5, 30, 60].map((s) => {
    const d = f.lookbacks?.[s]?.[5];
    return `${s}s: ${d?.status === "AVAILABLE" ? `bid ${numeric(d.bid_change)} / ask ${numeric(d.ask_change)}` : "INSUFFICIENT_HISTORY"}`;
  }).join(" · "));
  const p = f.lookbacks?.[60]?.[5] || {};
  text(`flow-persistence-${m}`, p.bid_heavy_fraction == null ? "INSUFFICIENT_HISTORY" : `60s · BID_HEAVY ${numeric(p.bid_heavy_fraction * 100, "%")} / ASK_HEAVY ${numeric(p.ask_heavy_fraction * 100, "%")} / BALANCED ${numeric(p.balanced_fraction * 100, "%")}`);
  text(`flow-last-${m}`, `${numeric(f.latest_trade?.price)} × ${numeric(f.latest_trade?.size)} · no inferred execution count`);
  text(`flow-volume-${m}`, `Reported ${numeric(f.volume?.value)} · ${typeof f.volume?.change === "number" ? `traded in the last 60 s ${numeric(f.volume.change)}` : "change UNAVAILABLE"} · ${f.volume?.status || "UNAVAILABLE"}`);
  if ($(`flow-detail-${m}`).open) text(`flow-meta-${m}`, JSON.stringify({version: f.version, calculated_at: f.at, last_receipt: feed.last_receipt, last_field_change: feed.last_field_change, last_contact: feed.last_contact, cadence, available_fields: f.available_fields, recording: rec, matched_price_changes: f.lookbacks}, null, 2));
}

function depth(m, d) {
  if (route !== "markets") return;
  const expires = (d.valid_until || 0) * 1000;
  depthExpiry.set(m, d.fresh ? expires : 0);
  text(`l2-${m}`, `L2 ${display(d.status)}${d.target_pre_seconds ? ` · ${Math.floor(d.pre_seconds || 0)} / ${d.target_pre_seconds}s pre-context` : ""}`);
  const fresh = d.fresh && serverNow() <= expires;
  text(`depth-note-${m}`, `${display(d.reason) ? display(d.reason) + " · " : ""}${fresh ? `${d.bids?.length || 0} bid / ${d.asks?.length || 0} ask levels · last depth receipt ${time(d.last_receipt)}` : "No current ladder"}`);
  const rows = Array.from({length: 10}, (_, i) => ({bid: fresh ? d.bids?.[i] : null, ask: fresh ? d.asks?.[i] : null}));
  const largest = Math.max(1, ...rows.flatMap((r) => [r.bid?.size || 0, r.ask?.size || 0]));
  rows.forEach(({bid, ask}, i) => {
    const row = $(`depth-${m}-${i}`);
    if (!row) return;
    const visible = Boolean(bid || ask);
    if (row.hidden === visible) row.hidden = !visible;
    const cells = row.children;
    // Sellers left, buyers right, the same way round as the imbalance bar.
    [[0, ask?.size], [2, ask?.price], [3, bid?.price], [5, bid?.size]].forEach(([j, v]) => {
      const value = String(v ?? "—");
      if (cells[j].textContent !== value) cells[j].textContent = value;
    });
    [[1, ask?.size], [4, bid?.size]].forEach(([j, v]) => {
      const width = `${(100 * (v || 0) / largest).toFixed(1)}%`;
      const bar = cells[j].firstElementChild;
      if (bar.style.width !== width) bar.style.width = width;
    });
  });
}

function svgNode(tag, attrs, label) {
  const node = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (label != null) node.textContent = label;
  return node;
}
function chart(m) {
  if (route !== "markets") return;
  const bars = m.chart || [], line = $(`line-${m.market}`), group = $(`markers-${m.market}`);
  text(`chart-empty-${m.market}`, bars.length ? "" : "Awaiting completed bars");
  if (!bars.length) {
    line.setAttribute("points", "");
    for (const id of [`axis-${m.market}`, `clocks-${m.market}`, `band-${m.market}`]) $(id).replaceChildren();
    $(`rv-${m.market}`).setAttribute("width", "0");
    chartGeometry.delete(m.market);
    return;
  }
  const start = Date.parse(bars[0].at), end = Date.parse(bars.at(-1).at) + 60000;
  const x = (t) => 50 + (580 * (t - start)) / Math.max(1, end - start);
  const inRange = (t) => t >= start && t <= end;
  const signature = JSON.stringify([bars, m.chart_context, (m.trades || []).map((t) => [t.entry_at, t.exit_at])]);
  if (chartGeometry.get(m.market) !== signature) {
    chartGeometry.set(m.market, signature);
    const prices = bars.map((b) => b.close), lo = Math.min(...prices), hi = Math.max(...prices), span = hi - lo || 1;
    const y = (p) => 176 - ((p - lo) * 160) / span;
    line.setAttribute("points", bars.map((b) => `${x(Date.parse(b.at) + 60000).toFixed(1)},${y(b.close).toFixed(1)}`).join(" "));
    const axis = $(`axis-${m.market}`);
    axis.replaceChildren(
      ...[hi, (hi + lo) / 2, lo].flatMap((p) => [svgNode("line", {x1: 50, x2: 630, y1: y(p), y2: y(p), class: "grid"}), svgNode("text", {x: 44, y: y(p) + 4, "text-anchor": "end"}, Number(p.toPrecision(6)))]),
      svgNode("text", {x: 50, y: 204, "text-anchor": "start"}, `${hhmm(start)} NY`),
      svgNode("text", {x: 630, y: 204, "text-anchor": "end"}, `${hhmm(end)} NY`),
    );
    const marks = (m.chart_context?.clocks || []).filter((c) => inRange(Date.parse(c.at)));
    $(`clocks-${m.market}`).replaceChildren(...marks.flatMap((c) => {
      const cx = x(Date.parse(c.at)), closed = ["CLOSED", "BREAK", "HALT", "SUSPENDED"].includes(String(c.session).toUpperCase());
      return [svgNode("line", {x1: cx, x2: cx, y1: 10, y2: 180, class: closed ? "closed" : ""}), svgNode("text", {x: cx, y: 192, "text-anchor": "middle"}, hhmm(c.at))];
    }));
    const rv = $(`rv-${m.market}`), rvWindow = m.chart_context?.rv_window;
    if (rvWindow) {
      const a = x(Math.max(start, Date.parse(rvWindow[0]))), b = x(Date.parse(rvWindow[1]));
      rv.setAttribute("x", a.toFixed(1)); rv.setAttribute("width", Math.max(0, b - a).toFixed(1));
    } else rv.setAttribute("width", "0");
    $(`band-${m.market}`).replaceChildren(...(m.trades || []).filter((t) => t.entry_at).map((t) => {
      const a = x(Math.max(start, Date.parse(t.entry_at))), b = x(Math.min(end, Date.parse(t.exit_at)));
      return svgNode("rect", {x: a, y: 10, width: Math.max(2, b - a), height: 170}, null);
    }));
  }
  const wanted = new Set();
  const markers = [
    ...(m.signals || []).map((s) => ({id: `signal-${s.id}`, at: s.signal_at, label: `Opportunity · ${display(s.decision)} · ${display(s.reason)}`, kind: "signal"})),
    ...(m.trades || []).filter((t) => t.entry_at).map((t) => ({id: `fill-${t.id}`, at: t.entry_at, label: `Paper fill · ${t.basis || "basis unverified"}`, kind: "fill"})),
  ];
  for (const t of markers) {
    const at = Date.parse(t.at);
    if (!inRange(at)) continue;
    wanted.add(t.id);
    let mark = [...group.children].find((n) => n.dataset.key === t.id);
    if (!mark) {
      mark = svgNode("line", {y1: 6, y2: 184});
      mark.dataset.key = t.id;
      mark.dataset.kind = t.kind;
      mark.append(svgNode("title", {}, `${t.label} ${time(t.at)}`));
      group.append(mark);
    }
    mark.setAttribute("x1", x(at));
    mark.setAttribute("x2", x(at));
  }
  [...group.children].filter((n) => !wanted.has(n.dataset.key)).forEach((n) => n.remove());
}

function smileChart(m) {
  const svg = $(`smile-${m.market}`), data = m.smile;
  if (!svg) return;
  const right = m.market === "CL" ? "call" : "put";
  const verified = data && data.scaling !== "PROVIDER_NATIVE_UNVERIFIED";
  const points = (data?.strikes || []).map((s) => ({strike: s.strike, vol: verified ? s[right]?.iv : s[right]?.mid_volatility ?? s.mid_volatility_pct, spread: s[right]?.iv_minus_model, oi: s[right]?.open_interest})).filter((p) => typeof p.vol === "number");
  const {o} = optionChoice(m);
  const signature = JSON.stringify([points, o?.identity?.strike, data?.mid_strike_price]);
  if (svg.dataset.signature === signature) return;
  svg.dataset.signature = signature;
  if (points.length < 2) {
    svg.replaceChildren(svgNode("text", {x: 150, y: 48, "text-anchor": "middle"}, data ? "Chain window has too few strikes with volatility" : "No options chain received"));
    text(`smile-note-${m.market}`, "");
    return;
  }
  const lo = Math.min(...points.map((p) => p.strike)), hi = Math.max(...points.map((p) => p.strike));
  const vlo = Math.min(...points.map((p) => p.vol)), vhi = Math.max(...points.map((p) => p.vol));
  const x = (v) => 8 + (284 * (v - lo)) / Math.max(1e-9, hi - lo), y = (v) => 60 - (48 * (v - vlo)) / Math.max(1e-9, vhi - vlo);
  const oiMax = Math.max(1, ...points.map((p) => p.oi || 0));
  const nodes = points.map((p) => svgNode("rect", {x: x(p.strike) - 3, y: 86 - (20 * (p.oi || 0)) / oiMax, width: 6, height: (20 * (p.oi || 0)) / oiMax, class: "oi"}));
  nodes.push(svgNode("polyline", {fill: "none", class: "vol", points: points.map((p) => `${x(p.strike).toFixed(1)},${y(p.vol).toFixed(1)}`).join(" ")}));
  if (typeof data.mid_strike_price === "number" && data.mid_strike_price >= lo && data.mid_strike_price <= hi) nodes.push(svgNode("line", {x1: x(data.mid_strike_price), x2: x(data.mid_strike_price), y1: 4, y2: 86, class: "underlying"}));
  const strike = o?.identity?.strike;
  if (typeof strike === "number" && strike >= lo && strike <= hi) nodes.push(svgNode("circle", {cx: x(strike), cy: y(points.find((p) => p.strike === strike)?.vol ?? vlo), r: 3.5, class: "selected"}));
  nodes.push(svgNode("text", {x: 4, y: 9}, numeric(vhi)), svgNode("text", {x: 4, y: 68}, numeric(vlo)));
  svg.replaceChildren(...nodes);
  const atTicket = points.find((p) => p.strike === strike);
  const comparison = verified
    ? `annual IV (${data.scaling.toLowerCase()} verified) · frozen model σ ${numeric(m.model_sigma)}${typeof atTicket?.spread === "number" ? ` · IV − model σ at ticket strike ${atTicket.spread >= 0 ? "+" : ""}${atTicket.spread.toFixed(3)}` : ""}`
    : `provider units unverified · frozen model σ ${numeric(m.model_sigma)} (annualised RV15)`;
  text(`smile-note-${m.market}`, `${right} IV by strike, expiry ${String(data.expiry || "").slice(0, 10)} · bars: open interest · dashed: underlying · dot: ticket strike · ${comparison}`);
}
function spark(id, series, {min, max, zero} = {}) {
  const svg = $(id);
  if (!svg) return;
  const values = series.flat().filter((v) => typeof v === "number");
  if (!values.length) { svg.replaceChildren(svgNode("text", {x: 150, y: 34, "text-anchor": "middle"}, "No retained samples")); return; }
  const lo = min ?? Math.min(...values), hi = max ?? Math.max(...values), span = hi - lo || 1;
  const width = Math.max(1, series[0].length - 1);
  const y = (v) => 54 - ((v - lo) * 48) / span;
  const nodes = series.map((line, k) => svgNode("polyline", {class: `s${k}`, fill: "none", points: line.map((v, i) => typeof v === "number" ? `${(300 * i / width).toFixed(1)},${y(v).toFixed(1)}` : null).filter(Boolean).join(" ")}));
  if (zero) nodes.unshift(svgNode("line", {x1: 0, x2: 300, y1: y(0), y2: y(0), class: "zero"}));
  nodes.push(svgNode("text", {x: 298, y: 10, "text-anchor": "end"}, numeric(values.at(-1))));
  svg.replaceChildren(...nodes);
}
function heatmap(m, data) {
  const canvas = $(`heatmap-${m}`), rows = data.series || [];
  const note = (v) => text(`heatmap-note-${m}`, v);
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  canvas.dataset.columns = String(rows.length);
  const ticks = rows.flatMap((r) => [...r.bid, ...r.ask].map(([t]) => t));
  if (!ticks.length) { note(data.status === "CONTRACT_NOT_VERIFIED" ? "Needs a verified futures contract" : "No retained depth samples yet"); return; }
  const last = rows.at(-1), center = ((last.bid[0]?.[0] ?? ticks[0]) + (last.ask[0]?.[0] ?? ticks[0])) / 2;
  const lo = Math.max(Math.min(...ticks), Math.floor(center - 20)), hi = Math.min(Math.max(...ticks), Math.ceil(center + 20));
  const levels = Math.max(1, hi - lo + 1), w = canvas.width / rows.length, h = canvas.height / levels;
  const largest = Math.max(1, ...rows.flatMap((r) => [...r.bid, ...r.ask].map(([, s]) => s || 0)));
  rows.forEach((r, i) => {
    for (const [side, color] of [["bid", "111, 207, 151"], ["ask", "235, 138, 122"]]) {
      for (const [tick, size] of r[side]) {
        if (tick < lo || tick > hi || !size) continue;
        ctx.fillStyle = `rgba(${color}, ${Math.min(1, 0.15 + 0.85 * Math.sqrt(size / largest)).toFixed(3)})`;
        ctx.fillRect(i * w, (hi - tick) * h, Math.ceil(w), Math.ceil(h));
      }
    }
  });
  const tick = data.tick_size;
  note(`${rows.length} × ${data.bucket_seconds}s buckets · ${levels} price levels${tick ? ` (${numeric(lo * tick)}–${numeric(hi * tick)})` : ""} · brighter = larger displayed size · ${data.semantics}`);
}
async function bookHistory(m, signal) {
  const panel = $(`book-history-${m}`);
  if (route !== "markets" || !panel?.open) return;
  const data = await request(`/api/market/${m}/book`, {signal});
  if (signal.aborted) return;
  const rows = data.series || [];
  spark(`spark-spread-${m}`, [rows.map((r) => r.spread_ticks)]);
  spark(`spark-size-${m}`, [rows.map((r) => r.bid[0]?.[1]), rows.map((r) => r.ask[0]?.[1])]);
  spark(`spark-imbalance-${m}`, [rows.map((r) => r.imbalance5)], {min: -1, max: 1, zero: true});
  heatmap(m, data);
}

function renderStatus(s) {
  limits = s.limits || limits;
  paused = s.paused;
  if (typeof s.server_time === "number") serverOffset = s.server_time * 1000 - Date.now();
  if (s.next_clock) { nextClock = s.next_clock; countdown(); }
  text("env-chip", `${s.data_environment || "Data unverified"} data`);
  text("exec-chip", `Execution ${s.execution_mode || "DISABLED"}`);
  text("entries", s.paused ? "Paused" : s.armed ? "Armed · gates apply" : "Unarmed");
  text("connection", s.connected ? (s.reconciled ? "Connected · reconciled" : "Connected · reconciliation required") : "Disconnected");
  $("connection").classList.toggle("bad", !s.connected);
  text("pause", paused ? "Resume entries" : "Pause entries");
  const exceptions = Object.values(s.management_problems || {}).join(" · ");
  text("execution-warning", exceptions || "No reported exposure exceptions");
  text("global-warning", exceptions || s.problem || s.l2_recording?.paused_reason || "");
  const primary = s.session?.TradeLevel === "FullTradingAndChat";
  const live = (s.setup || []).find((item) => item.key === "realtime")?.done;
  text("realtime-chip", primary ? (live ? "Real-time ON" : "Real-time ON · Saxo still delayed") : "Real-time OFF");
  $("realtime-chip").classList.toggle("bad", !(primary && live));
  text("primary-state", !primary ? "OFF: prices are delayed. Click to give SLRNO Saxo's real-time slot."
    : live ? "ON: SLRNO holds Saxo's real-time slot and prices are real-time."
    : "ON, but Saxo still sends delayed prices. In SaxoTraderGO check My Profile → Other → Open API Access is enabled; that login turns this OFF, so click again after it.");
  text("auth-state", `OAuth: ${s.oauth || "UNVERIFIED"}${s.oauth_problem ? ` (${s.oauth_problem})` : ""} · stream: ${s.connected ? "connected" : "disconnected"} · session: ${s.session?.TradeLevel || "UNVERIFIED"}`);
  const alerts = s.alerts || {};
  text("events-state", `Saxo order/position events: ${display(s.activity_events || "NOT_SUBSCRIBED").toLowerCase()}${s.closed_positions_problem ? ` · closed positions: ${s.closed_positions_problem}` : ""}${s.calendar_problem ? ` · event calendar: ${s.calendar_problem}` : ""}`);
  text("alerts-state", `Alerts: ${alerts.enabled ? `enabled · ${alerts.sent || 0} sent${alerts.last_error ? ` · last error ${alerts.last_error}` : ""}` : alerts.problem ? display(alerts.problem) : "not configured"}${(alerts.active || []).length ? ` · active: ${alerts.active.join(" · ")}` : ""}`);
  if (s.setup) renderSetup(s.setup);
  for (let i = 0; i < 4; i++) {
    const used = i < s.reserved_open_trades;
    const slot = $(`overview-slot-${i}`);
    if (slot) { slot.classList.toggle("used", used); slot.title = `Slot ${i + 1} · ${used ? "reserved / open" : "available"}`; }
    text(`trade-slot-${i}`, `Slot ${i + 1} · ${used ? "reserved / open" : "available"}`);
  }
  text("capacity", `${s.reserved_open_trades ?? "—"} / ${limits.slots ?? "—"} in use`);
  text("allocation-policy", `${money(limits.per_trade_gbp)} all-in per trade · one contract · concurrent ceiling, not a daily loss limit`);
  text("allocation", `${money(s.allocation_pennies / 100)} / ${money(limits.allocation_gbp)}`);
}
function renderSetup(items) {
  const list = $("setup-list");
  const required = items.filter((i) => !i.optional), done = required.filter((i) => i.done).length;
  text("setup-progress", `${done} of ${required.length} required steps done`);
  items.forEach((item, i) => {
    let li = list.children[i];
    if (!li) { li = document.createElement("li"); li.append(document.createElement("strong"), document.createElement("span")); list.append(li); }
    const cls = `${item.done ? "done" : "todo"}${item.optional ? " optional" : ""}`;
    if (li.className !== cls) li.className = cls;
    const [label, detail] = li.children;
    const labelText = `${item.label}${item.optional ? " (optional)" : ""}`;
    if (label.textContent !== labelText) label.textContent = labelText;
    if (detail.textContent !== item.detail) detail.textContent = item.detail;
  });
  // The overview hides a completed checklist and shows it again if a step regresses.
  if (route === "overview") show("setup", done !== required.length);
}
function render(d) {
  const s = d.system, p = d.pnl || {skip_reasons: {}};
  p.skip_reasons ||= {};
  d.markets ||= [];
  if (d.account) renderAccount(d.account);
  renderStatus(s);
  if (d.pnl) {
    text("realised", money(p.realised_net_gbp));
    const saxo = p.broker_reported || {};
    const reported = saxo.count ? ` · Saxo reports ${numeric(saxo.closed_profit_loss_base)} ${saxo.currency || "account currency"} closed P&L (${saxo.count} position${saxo.count === 1 ? "" : "s"})` : "";
    text("pnl-note", (p.provisional_closed ? `${p.provisional_closed} closed trade(s) provisional · missing costs/FX` : display(p.basis || "Paper evidence unavailable")) + reported);
    const vals = d.markets.flatMap((m) => m.trades || []).filter((t) => t.quantity > 0).map((t) => t.valuation);
    text("unrealised", vals.length && vals.every((v) => v.fresh && v.value_gbp != null) ? money(vals.reduce((a, v) => a + v.value_gbp, 0)) : vals.length ? "Unavailable" : "—");
    text("valuation-note", vals.length ? "Bid estimate before commissions" : "No open paper positions");
    text("evidence", `${p.opportunities ?? 0} opportunities · ${p.eligible_trades ?? 0} eligible · ${p.skip_reasons.MINIMUM_CONTRACT_COST_EXCEEDS_BUDGET || 0} budget skips · ${p.skip_reasons.SKIP_CAPACITY_FULL || 0} capacity skips · ${p.fills ?? 0} executions · ${p.win_rate == null ? "win rate unavailable" : `${(p.win_rate * 100).toFixed(1)}% win rate`} (${p.wins ?? 0}/${p.closed_with_complete_costs ?? 0} closed with complete costs)`);
  }
  for (const m of d.markets) {
    if (!markets.includes(m.market)) continue;
    renderGates(m);
    text(`state-${m.market}`, display(m.strategy_state));
    $(`card-${m.market}`).dataset.state = m.strategy_state || "";
    text(`block-${m.market}`, firstBlock(m));
    $(`block-${m.market}`).title = m.block_reason || "";
    if (route !== "markets") {
      text(`contract-${m.market}`, m.contract || "Contract awaiting verification");
      const trades = m.trades || [];
      show(`position-${m.market}`, trades.length > 0);
      text(`position-${m.market}`, trades.map((t) => `${display(t.state)} · ${t.quantity} contract · exit ${hhmm(t.exit_at)} NY`).join(" · "));
      show(`candidate-${m.market}`, Boolean(m.candidate_uic));
      text(`candidate-${m.market}`, m.candidate_uic ? `Candidate strike UIC ${m.candidate_uic}` : "");
      const events = m.events || [];
      show(`events-${m.market}`, events.length > 0);
      text(`events-${m.market}`, events.map((e) => `${e.name} ${hhmm(e.at)} NY · ${display(e.relation).toLowerCase()}`).join(" · "));
      text(`data-${m.market}`, `Quote ${display(m.data_status).toLowerCase()} · ${m.last_receipt ? time(new Date(m.last_receipt * 1000).toISOString()) : "no receipt"}`);
      continue;
    }
    quoteReceipts.set(m.market, Date.parse(m.l1?.last_receipt));
    text(`capability-${m.market}`, `${m.market} · ${m.l1?.status || "UNVERIFIED"} · ${m.l2?.status || "L2_UNAVAILABLE"} · ${m.block_reason || "monitoring"}`);
    const q = m.l1?.quote || {}, sizes = m.l1?.sizes || {}, rec = m.recorder || {};
    text(`quote-${m.market}`, `Bid ${q.Bid ?? "—"} × ${sizes.bid ?? "—"} · Ask ${q.Ask ?? "—"} × ${sizes.ask ?? "—"} · spread ${m.l1?.spread != null ? numeric(m.l1.spread) : "—"} · delay ${m.l1?.delay_minutes ?? "unknown"} min`);
    text(`recorder-${m.market}`, `${rec.state || "UNAVAILABLE"} · ${Math.floor(rec.prehistory_seconds || 0)} / 900s prehistory${rec.reason ? " · " + display(rec.reason) : ""}`);
    text(`contract-${m.market}`, m.contract || "");
    text(`market-${m.market}`, display(m.market_status));
    text(`data-${m.market}`, `Bars ${display(m.data_status)}`);
    text(`l1-${m.market}`, `L1 ${display(m.l1?.status || "DISCONNECTED")}`);
    text(`direction-${m.market}`, m.direction);
    text(`conditions-${m.market}`, m.conditions?.rv15 != null ? `RV15 ${(m.conditions.rv15 * 100).toFixed(4)}% · completed bars` : "Waiting for required history");
    const pc = m.price_context;
    text(`price-context-${m.market}`, pc ? [pc.open != null && `open ${pc.open}`, pc.high != null && `high ${pc.high}`, pc.low != null && `low ${pc.low}`, pc.last_close != null && `prev close ${pc.last_close}`, pc.net_change != null && `change ${numeric(pc.net_change)}${pc.percent_change != null ? ` (${Number(pc.percent_change).toFixed(2)}%)` : ""}`, pc.open_interest != null && `OI ${pc.open_interest}`, pc.market_state && pc.market_state, pc.daily_range != null && `20-day avg range ${numeric(pc.daily_range)}`].filter(Boolean).join(" · ") || "Not reported" : "Not reported");
    const sessions = m.sessions_today || [];
    text(`sessions-${m.market}`, sessions.length ? sessions.map((x) => `${hhmm(x.start)}–${hhmm(x.end)} ${x.state}`).join(" · ") + " (NY)" : "Session schedule unverified");
    const events = m.events_today || [];
    text(`events-today-${m.market}`, events.length ? events.map((e) => `${hhmm(e.at)} ${e.name}`).join(" · ") + " (NY)" : "No scheduled releases listed");
    depth(m.market, m.l2 || {status: "DISABLED"});
    bookFlow(m.market, m.book_flow || {}, rec, m.identity || {environment: s.data_environment});
    optionContext(m);
    smileChart(m);
    const trades = m.trades || [];
    text(`position-${m.market}`, trades.length ? trades.map((t) => `${display(t.state)} · ${t.quantity} contract · exit ${hhmm(t.exit_at)} NY${t.quantity ? ` · bid P&L ${t.valuation.fresh ? money(t.valuation.value_gbp) : "unavailable"}` : ""}`).join(" | ") : "No pending or open paper trade");
    if ($(`raw-state-${m.market}`).open) text(`details-${m.market}`, JSON.stringify({...m.details, l1: m.l1, l2: m.l2, recorder: m.recorder, capabilities: m.capabilities, gates: m.gates, chart_context: m.chart_context, sessions_today: m.sessions_today, trades: m.trades}, null, 2));
    chart(m);
  }
}
function renderSystem(d) {
  render({system: d});
  const a = d.market_data || {}, l = d.l2_recording || {};
  text("api-lines", `${a.owned_lines ?? "—"} / ${a.app_budget ?? "—"} subscriptions`);
  text("api-allowance", `Saxo session: ${d.session?.TradeLevel || "UNVERIFIED"}`);
  text("api-depth", `${(d.markets || []).filter((m) => m.capabilities?.l2?.status === "L2_AVAILABLE").length} / 5 markets with received depth`);
  text("api-options", `${a.option_lines ?? 0} / ${a.option_budget ?? "—"} regular option quote subscriptions`);
  text("api-pacing", Object.entries(a.rate_limits || {}).map(([k, v]) => `${k.replace("x-ratelimit-", "")} ${v}`).join(" · ") || "No rate-limit headers received yet");
  text("api-queue", `${a.rest_queue ?? 0} / ${a.rest_queue_limit ?? "—"} queued REST requests`);
  text("api-storage", `${((l.disk_bytes || 0) / 1048576).toFixed(1)} / ${((l.disk_limit || 0) / 1048576).toFixed(0)} MiB stored`);
  text("api-gaps", `${l.recording_gaps ?? 0} recording gaps · ${l.writer_queue ?? 0} queued batches`);
  text("api-error", l.paused_reason || "No reported recording problem");
  for (const m of d.markets || []) text(`capability-${m.market}`, `${m.market} · ${display(m.problem) || "Connected"}`);
  if ($("system-detail").open) text("system-json", JSON.stringify(d, null, 2));
}

async function request(path, {method = "GET", signal, timeout = 8000} = {}) {
  const controller = new AbortController();
  const cancel = () => controller.abort();
  signal?.addEventListener("abort", cancel, {once: true});
  if (signal?.aborted) controller.abort();
  const timer = setTimeout(() => controller.abort("timeout"), timeout);
  try {
    const response = await fetch(path, {method, cache: "no-store", credentials: "same-origin", headers: {Accept: "application/json"}, signal: controller.signal});
    const result = await response.json().catch((error) => { if (controller.signal.aborted) throw error; return null; });
    if (!response.ok) throw Error((typeof result?.detail === "string" && result.detail) || result?.error || `Request failed (${response.status})`);
    if (result == null) throw Error("The server returned an unreadable response");
    return result;
  } catch (error) {
    if (controller.signal.reason === "timeout") throw Error("Request timed out; retrying on the next refresh");
    if (controller.signal.aborted) throw new DOMException("Request cancelled", "AbortError");
    throw error instanceof TypeError ? Error("Connection unavailable; checking again shortly") : error;
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener("abort", cancel);
  }
}
function renderAccount(a) {
  accountSnapshot = a;
  text("account", `${a.label} · ${a.account} · ${a.currency || "Currency unavailable"}`);
  text("account-note", a.connection_note);
  const native = (v) => typeof v === "number" && a.currency ? `${NATIVE.format(v)} ${a.currency}` : "Unavailable";
  text("account-value", native(a.total_value));
  text("account-cash", native(a.cash_balance));
  text("account-available", native(a.cash_available_for_trading));
  text("account-freshness", `${a.status} · last successful update ${a.last_success_at != null ? time(new Date(a.last_success_at * 1000).toISOString()) : "Unavailable"}`);
  if ($("account-detail").open) text("account-json", JSON.stringify({basis: a.basis, ...a.details, problem: a.problem}, null, 2));
}
// Keyed rows keep their DOM nodes (focus, selection, open state) across refreshes.
function rows(id, items, columns, key, create = () => {}) {
  const body = $(id), known = new Map([...body.children].map((n) => [n.dataset.key, n]));
  for (const [index, item] of items.entries()) {
    const identity = String(item[key]);
    let node = known.get(identity); known.delete(identity);
    if (!node) { node = document.createElement("tr"); node.dataset.key = identity; columns.forEach(() => node.append(document.createElement("td"))); create(node, item); }
    columns.forEach((f, i) => { const next = String(f(item) ?? "Unavailable"); if (node.children[i].textContent !== next) node.children[i].textContent = next; });
    if (body.children[index] !== node) body.insertBefore(node, body.children[index] || null);
  }
  known.forEach((n) => n.remove());
}
function renderExecution(d) {
  render({system: d.system});
  rows("execution-trades", d.trades, [(t) => t.market, (t) => display(t.state), (t) => t.option.symbol || t.option.uic, (t) => t.quantity, (t) => money(t.allocation_pennies / 100), (t) => time(t.exit_at)], "id");
  rows("execution-orders", d.orders, [(o) => o.role, (o) => o.order_id, (o) => display(o.status), (o) => o.filled, (o) => o.remaining, (o) => time(o.deadline)], "reference");
  rows("execution-fills", d.fills, [(f) => time(f.at), (f) => f.con_id, (f) => f.side, (f) => f.quantity, (f) => f.price, (f) => f.commission == null ? "Unavailable" : `${f.commission} ${f.commission_currency || "currency unverified"}`], "exec_id");
  rows("execution-positions", d.positions, [(p) => p.con_id, (p) => p.quantity], "con_id");
  text("execution-empty", d.trades.length ? "Internal reservations include pending entries and open trades." : "No pending or open SLRNO trades.");
  if ($("execution-detail").open) text("execution-json", JSON.stringify(d, null, 2));
}
async function history(signal) {
  const q = new URLSearchParams({offset: String(offset)});
  for (const [key, id] of [["market", "market-filter"], ["day", "day-filter"], ["version", "version-filter"]]) if ($(id).value) q.set(key, $(id).value);
  q.set("sort", $("sort-filter").value);
  const d = await request(`/api/history?${q}`, {signal});
  if (signal.aborted) return;
  pageSize = d.page_size || pageSize;
  if (d.system) render({system: d.system});
  rows("history", d.rows, [(r) => time(r.signal_at), (r) => r.market, (r) => display(r.state || r.decision), (r) => display(r.reason) || "—", (r) => r.rule_version], "id", (tr, r) => {
    const b = document.createElement("button");
    b.textContent = "Inspect";
    b.onclick = () => { detailSignature = JSON.stringify(r); showDetail(r.id).catch(reportError); };
    tr.append(document.createElement("td"));
    tr.lastChild.append(b);
  });
  $("history-empty").hidden = d.rows.length > 0;
  $("prev").disabled = offset === 0;
  $("next").disabled = !d.has_more;
  text("page-number", `Page ${1 + offset / pageSize}`);
  // Evidence is refetched only when the selected row's summary changed.
  const selected = JSON.stringify(d.rows.find((r) => r.id === selectedIdentity) ?? null);
  if (selectedIdentity && $("trade-detail").open && selected !== detailSignature) {
    detailSignature = selected;
    await showDetail(selectedIdentity, false);
  }
}
function renderTimeline(steps) {
  const list = $("trade-timeline");
  list.replaceChildren(...(steps || []).map((s) => {
    const li = document.createElement("li");
    li.className = s.ok === true ? "ok" : s.ok === false ? "fail" : "neutral";
    const head = document.createElement("strong");
    head.textContent = s.title;
    const when = document.createElement("time");
    when.textContent = s.at ? time(s.at) : "";
    const body = document.createElement("span");
    body.textContent = display(s.detail) === s.detail ? s.detail : `${display(s.detail)} (${s.detail})`;
    li.append(head, when, body);
    return li;
  }));
}
async function showDetail(id, expand = true) {
  selectedIdentity = id;
  detailRequest?.abort();
  const controller = new AbortController(); detailRequest = controller;
  if (expand) $("trade-detail").open = true;
  const d = await request(`/api/detail?identity=${encodeURIComponent(id)}`, {signal: controller.signal});
  if (controller.signal.aborted || selectedIdentity !== id) return;
  renderTimeline(d.timeline);
  text("trade-json", JSON.stringify(d, null, 2));
}
function reportError(e) {
  if (e.name !== "AbortError") text("notice", `${e.message} · displayed values are stale`);
}
async function refresh(force = false) {
  if (document.hidden || controlPending || (pending && !force)) return;
  if (force) pending?.abort();
  const controller = new AbortController(); pending = controller;
  try {
    if (page === "trades") await history(controller.signal);
    else {
      let path = "/api/overview";
      if (route === "markets") path = `/api/market/${selectedMarket}?diagnostics=${$(`raw-state-${selectedMarket}`).open}`;
      if (page === "execution") path = "/api/execution";
      if (page === "system") path = `/api/system?diagnostics=${$("system-detail").open}`;
      const d = await request(path, {signal: controller.signal});
      if (controller.signal.aborted) return;
      if (page === "execution") renderExecution(d);
      else if (page === "system") {
        renderSystem(d);
        if ($("recordings-detail").open) {
          const recordings = await request("/api/recordings", {signal: controller.signal});
          if (controller.signal.aborted) return;
          const entries = [...(recordings.active || []), ...(recordings.completed || [])].slice(-132);
          const wanted = new Set(entries.map((r) => `recording-${r.segment}`));
          for (const node of [...$("recording-list").children]) if (!wanted.has(node.id)) node.remove();
          for (const rec of entries) {
            const id = `recording-${rec.segment}`;
            if (!$(id)) { const a = document.createElement("a"); a.id = id; a.href = `/api/recordings/${encodeURIComponent(rec.segment)}`; $("recording-list").append(a); }
            text(id, `${rec.state} · ${rec.key} · ${rec.segment.slice(0, 10)}`);
          }
        }
      } else {
        render(d);
        await bookHistory(selectedMarket, controller.signal);
      }
    }
    if (!controller.signal.aborted) {
      lastRefresh = new Date().toISOString(); text("notice", "");
      text("last-refresh", `Refreshed ${time(lastRefresh)}`);
    }
  } catch (e) { reportError(e); } finally { if (pending === controller) pending = null; }
}
$("saxo-connect").onsubmit = async (event) => {
  event.preventDefault();
  const button = event.currentTarget.querySelector("button"); button.disabled = true;
  text("oauth-notice", "");
  try { const result = await request("/oauth/saxo/start", {method: "POST", timeout: 15000}); window.location.assign(result.authorization_url); }
  catch (e) { text("oauth-notice", e.message); button.disabled = false; }
};
$("pause").onclick = async () => {
  pending?.abort(); controlPending = true; $("pause").disabled = true;
  try { const result = await request(`/api/entries/${paused ? "resume" : "pause"}`, {method: "POST"}); renderStatus(result); text("notice", result.paused ? "Server confirmed: entries paused" : "Server confirmed: entries resumed; readiness gates still apply"); }
  catch (e) { reportError(e); } finally { controlPending = false; $("pause").disabled = false; }
};
// Confirmed on the page with a second tap: some phone browsers silently block window.confirm (2026-10-01).
let primaryConfirmUntil = 0;
$("primary-session").onclick = async () => {
  if (Date.now() > primaryConfirmUntil) {
    primaryConfirmUntil = Date.now() + 5000;
    text("primary-session", "Tap again to confirm (SaxoTraderGO may go delayed)");
    setTimeout(() => { if (Date.now() > primaryConfirmUntil) text("primary-session", "Use real-time in SLRNO"); }, 5100);
    return;
  }
  primaryConfirmUntil = 0;
  text("primary-session", "Use real-time in SLRNO");
  pending?.abort(); controlPending = true; $("primary-session").disabled = true;
  try { renderStatus(await request("/api/session/primary", {method: "POST"})); text("notice", "Server confirmed: real-time requested; price streams renew"); }
  catch (e) { reportError(e); } finally { controlPending = false; $("primary-session").disabled = false; }
};
for (const id of ["market-filter", "day-filter", "version-filter", "sort-filter"])
  $(id).addEventListener("change", () => { offset = 0; selectedIdentity = ""; detailRequest?.abort(); text("trade-json", "Select an opportunity to view its evidence"); $("trade-timeline").replaceChildren(); refresh(true); });
$("prev").onclick = () => { offset = Math.max(0, offset - pageSize); refresh(true); };
$("next").onclick = () => { offset += pageSize; refresh(true); };
document.addEventListener("visibilitychange", () => { if (document.hidden) { pending?.abort(); detailRequest?.abort(); } else refresh(true); });
document.querySelectorAll("details").forEach((el) => el.addEventListener("toggle", () => {
  if (el.id === "account-detail") { if (accountSnapshot) renderAccount(accountSnapshot); }
  else if (el.open && el.id !== "setup-detail" && el.id !== "trade-detail") refresh(true); // Inspect fetches its own evidence
}));
function countdown() {
  if (!nextClock) return;
  const left = Math.max(0, Math.round((Date.parse(nextClock) - serverNow()) / 1000));
  const hours = Math.floor(left / 3600), minutes = Math.floor((left % 3600) / 60), seconds = left % 60;
  const span = hours ? `${hours}h ${String(minutes).padStart(2, "0")}m` : `${minutes}:${String(seconds).padStart(2, "0")}`;
  text("countdown", `Next clock ${hhmm(nextClock)} NY · in ${span}`);
}
setInterval(() => {
  text("clock", `${time(new Date().toISOString())} · London`);
  countdown();
  const at = serverNow();
  if (accountSnapshot?.status === "Current" && at > accountSnapshot.valid_until * 1000) renderAccount({...accountSnapshot, status: "Stale"});
  if (lastRefresh && Date.now() - Date.parse(lastRefresh) > 15000) text("last-refresh", `Stale · last refresh ${time(lastRefresh)}`);
  for (const [m, expires] of depthExpiry) if (expires && at > expires) depth(m, {status: "L2_UNAVAILABLE", reason: "SUBSCRIPTION_HEALTH_EXPIRED"});
  for (const [m, expires] of flowExpiry) if (expires && at > expires) bookFlow(m, {status: "UNAVAILABLE", quality_flags: ["SUBSCRIPTION_HEALTH_EXPIRED"]});
  for (const [m, received] of quoteReceipts) if (!received || at - received > limits.quote_max_age_seconds * 1000) text(`l1-${m}`, "L1 STALE OR MISSING");
}, 1000);
setInterval(() => refresh(), 5000);

for (let i = 0; i < 4; i++) {
  const slot = document.createElement("div"); slot.id = `trade-slot-${i}`;
  $("trade-slots").append(slot);
  const overview = document.createElement("div"); overview.id = `overview-slot-${i}`;
  $("overview-slots").append(overview);
}
for (const m of markets) {
  const row = document.createElement("p"); row.id = `capability-${m}`; $("capability-list").append(row);
}
if (route === "markets") $("markets").classList.add("single-market");
const linkedMarket = new URLSearchParams(location.search).get("market");
if (markets.includes(linkedMarket)) selectedMarket = linkedMarket;
$("selected-market").value = selectedMarket;
function selectMarket() {
  selectedMarket = $("selected-market").value;
  sessionStorage.setItem("slrno-market", selectedMarket);
  for (const market of markets) $(`card-${market}`).hidden = route === "markets" && market !== selectedMarket;
  refresh(true);
}
$("selected-market").addEventListener("change", selectMarket);
selectMarket();
