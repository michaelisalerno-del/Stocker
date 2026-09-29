"use strict";
const $ = (id) => document.getElementById(id);
const markets = ["CL", "GC", "NG", "NQ", "SI"];
const names = {
  CL: "Crude oil",
  GC: "Gold",
  NG: "Natural gas",
  NQ: "Nasdaq 100",
  SI: "Silver",
};
const depthReceipts = new Map();
const quoteReceipts = new Map();
const flowExpiry = new Map();
const route = location.pathname.slice(1) || "overview";
const page = ["opportunities", "trades"].includes(route) ? "trades" : route === "markets" ? "overview" : route;
let selectedMarket = sessionStorage.getItem("slrno-market") || "CL";
if (!markets.includes(selectedMarket)) selectedMarket = "CL";
let offset = 0,
  paused = false,
  pending = null,
  selectedIdentity = "";
let detailRequest = null, controlPending = false, lastRefresh = null, accountSnapshot = null;
const chartGeometry = new Map();
let limits = {};
const money = (v) =>
  v == null
    ? "Unavailable"
    : new Intl.NumberFormat("en-GB", {
        style: "currency",
        currency: "GBP",
      }).format(v);
const time = (v, zone = "Europe/London") =>
  v
    ? new Intl.DateTimeFormat("en-GB", {
        timeZone: zone,
        day: "2-digit",
        month: "short",
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      }).format(new Date(v))
    : "—";
const text = (id, v) => {
  const el = $(id),
    next = String(v ?? "—");
  if (el && el.textContent !== next) el.textContent = next;
};
const explanations = {
  REFERENCE_CONTRACT_SELECTION_REQUIRED: "Select and verify the Saxo futures contract",
  LISTED_PRODUCT_AND_DELTA_TOLERANCE_UNAPPROVED: "Option product and selection approval required",
  MINIMUM_CONTRACT_COST_EXCEEDS_BUDGET: "One contract, including entry and reserved exit costs, exceeds the per-trade ceiling",
  INCOMPLETE_COMPLETED_HISTORY: "Waiting for the final completed-minute bar",
  STALE_SIGNAL_NO_REPLAY: "The entry deadline passed before this opportunity was ready",
  EXECUTION_DISABLED: "Paper execution is disabled", PAPER_DISARMED: "Paper entries are not armed",
  SKIP_CAPACITY_FULL: "All four trade slots are occupied or reserved",
  RECONCILIATION_REQUIRED: "Broker reconciliation is required", ENTRIES_PAUSED: "New entries are paused",
  GC_LISTED_EXECUTION_RULE_UNAPPROVED: "Gold is monitor-only; execution is not approved",
  ORDER_STATUS_UNCERTAIN: "Broker order status needs reconciliation",
  EXPOSURE_REQUIRES_RECONCILIATION: "Exposure awaiting reconciliation",
  AUTHENTICATION_REQUIRED: "Connect the configured Saxo account",
};
const display = (v) => explanations[v] || String(v || "").replaceAll("_", " ");
for (const name of ["overview", "trades", "execution", "system"])
  $(name).hidden = name !== page;
text(
  "title",
  page === "trades"
    ? "Trades & signal history"
    : page === "system"
      ? "System"
      : page === "execution" ? "Paper execution" : route === "markets" ? "Market detail" : "Futures overview",
);
document.querySelectorAll("nav a").forEach((a) => {
  if (a.pathname === location.pathname) a.setAttribute("aria-current", "page");
});
// Stable nodes: updates preserve focus, expanded cards and scroll offsets.
for (const m of markets) {
  const card = document.createElement("article");
  card.className = "market";
  card.id = `card-${m}`;
  card.innerHTML = route !== "markets" ? `<div class="card-head"><h2>${m} <span>${names[m]}</span></h2><strong id="state-${m}"></strong></div><p id="contract-${m}"></p><p class="block" id="block-${m}"></p><p id="position-${m}"></p><p id="candidate-${m}"></p><div class="next"><span>Next decision · NY</span><strong id="next-${m}"></strong></div><small id="data-${m}"></small><a href="/markets?market=${m}">Market detail</a>` : `<div class="card-head"><div><h2>${m} <span>${names[m]}</span></h2><p id="contract-${m}"></p></div><span class="state" id="state-${m}"></span></div>
 <div class="status-line"><span id="market-${m}"></span><span id="data-${m}"></span></div>
 <div class="feed-line"><span id="l1-${m}"></span><span id="l2-${m}"></span></div>
 <svg class="chart" viewBox="0 0 420 92" role="img" aria-label="${m} completed underlying prices"><path class="chart-grid" d="M0 30H420 M0 62H420"/><polyline id="line-${m}" fill="none"/><g id="markers-${m}"></g><text id="chart-empty-${m}" x="210" y="48" text-anchor="middle">Awaiting completed bars</text></svg>
 <div class="condition"><span id="direction-${m}"></span><strong id="conditions-${m}"></strong></div>
 <p class="quote" id="quote-${m}"></p><p class="recorder-line" id="recorder-${m}"></p><p class="block" id="block-${m}"></p><div class="position" id="position-${m}"></div>
 <div class="next"><span>Next clock · NY</span><strong id="next-${m}"></strong></div>
 <p class="diagnostic" id="diagnostic-${m}"></p><details><summary>Contract, rule & research depth</summary>
 <div class="depth"><p>Research only — does not affect trades</p><small id="depth-note-${m}"></small>
 <table aria-label="${m} displayed depth"><thead><tr><th>Bid size</th><th>Bid</th><th>Ask</th><th>Ask size</th></tr></thead>
 <tbody>${Array.from({ length: 20 }, (_, i) => `<tr id="depth-${m}-${i}" hidden><td></td><td></td><td></td><td></td></tr>`).join("")}</tbody></table></div>
 <pre id="details-${m}"></pre></details>`;
  $("markets").append(card);
  if (route === "markets") {
    const optionPanel = document.createElement("section");
    optionPanel.className = "book-flow option-context";
    optionPanel.id = `option-context-${m}`;
    optionPanel.innerHTML = `<h3>Option context</h3>
      <p id="option-identity-${m}"></p><p id="option-deadline-${m}"></p>
      <p id="option-quote-${m}"></p><p id="option-analytics-${m}"></p>
      <p id="option-volume-${m}"></p><p id="option-underlying-${m}"></p>
      <p id="option-cost-${m}"></p><p id="option-coverage-${m}"></p>
      <details><summary>Greeks, assumptions and contract history</summary>
      <p>Provider analytics are unverified context. Chain indications and latest trades are not executable prices.</p>
      <pre id="option-meta-${m}"></pre></details>`;
    card.insertBefore(optionPanel, card.querySelector("details"));
    const panel = document.createElement("section");
    panel.className = "book-flow";
    panel.id = `book-flow-${m}`;
    panel.innerHTML = `<h3>BOOK FLOW — SAMPLED L2</h3>
      <p class="muted">Sampled order-book observations; not a complete execution tape.</p>
      <p id="flow-status-${m}"></p><p id="flow-feed-${m}"></p>
      <p id="flow-spread-${m}"></p>
      <div class="flow-table"><table aria-label="${m} book-flow depth totals"><thead><tr><th>Levels</th><th>Bid depth</th><th>Ask depth</th><th>Imbalance</th><th>Order imbalance</th><th>Observed state</th></tr></thead>
      <tbody>${[1,3,5,10].map(n => `<tr id="flow-depth-${m}-${n}"><th>${n}</th><td></td><td></td><td></td><td></td><td></td></tr>`).join("")}</tbody></table></div>
      <p id="flow-mid-${m}"></p><p id="flow-changes-${m}"></p><p id="flow-persistence-${m}"></p>
      <p id="flow-last-${m}"></p><p id="flow-volume-${m}"></p>
      <details id="flow-detail-${m}"><summary>Coverage, fields and observation times</summary><pre id="flow-meta-${m}"></pre></details>`;
    card.insertBefore(panel, optionPanel);
    panel.append(card.querySelector(".depth"));
  }
}
const numeric = (v, suffix = "") => typeof v === "number" && Number.isFinite(v) ? `${Number(v.toFixed(4))}${suffix}` : "UNAVAILABLE";
function optionContext(m) {
  if (route !== "markets") return;
  const context = m.option_context || {}, contracts = context.contracts || [];
  const held = m.trades?.[0], event = context.latest_event;
  const eventUic = event?.context?.identity?.uic;
  const uic = held?.option?.uic || (contracts.some(c => c.identity?.uic === eventUic) ? eventUic : context.candidate_uic);
  const o = contracts.find(c => c.identity?.uic === uic), id = o?.identity || {};
  const q = o?.quote || {}, cost = o?.costs || {};
  const obs = (name, rows) => {
    const r = rows?.[name];
    return r ? `${numeric(r.value)} · ${r.status} · age ${Math.floor(r.age_seconds)}s · effective ${r.effective_at || "unknown"}` : "MISSING";
  };
  const analytics = {...(o?.chain_analytics || {}), ...(o?.analytics || {})};
  text(`option-identity-${m.market}`, o ? `${held ? "Owned contract" : uic === eventUic ? "Event-selected contract" : "Current candidate"}: ${id.symbol || id.uic} · ${id.right} ${id.strike} · UIC ${id.uic} · underlying ${id.underlying_symbol || "unknown"} / ${id.underlying_uic}` : `No selected contract · ${context.problem || "UNAVAILABLE"}`);
  text(`option-deadline-${m.market}`, `Expiry ${id.expiry || "unknown"} · last trading ${id.last_trade_at || "UNVERIFIED"} · strategy exit ${held?.exit_at || (uic === eventUic && event?.context?.strategy_exit_at) || "set by event + 60 minutes"}`);
  text(`option-quote-${m.market}`, `Regular quote ${o?.quote_status || "MISSING"} · bid ${numeric(q.Bid)} × ${numeric(o?.sizes?.Bid)} (${q.PriceTypeBid || "unknown"}) · ask ${numeric(q.Ask)} × ${numeric(o?.sizes?.Ask)} (${q.PriceTypeAsk || "unknown"}) · spread ${numeric(typeof q.Ask === "number" && typeof q.Bid === "number" ? q.Ask-q.Bid : null)} · delay ${q.DelayedByMinutes ?? "unknown"} min`);
  text(`option-analytics-${m.market}`, `Provider delta ${obs("Greeks.Delta", analytics)} · IV ${obs(analytics["Greeks.MidVolatility"] ? "Greeks.MidVolatility" : "Greeks.MidVol", analytics)} · provider units / scaling unverified`);
  text(`option-volume-${m.market}`, `Option volume ${obs("PriceInfoDetails.Volume", analytics)} · OI ${obs("InstrumentPriceDetails.OpenInterest", analytics)}`);
  text(`option-underlying-${m.market}`, `Future volume ${obs("PriceInfoDetails.Volume", m.underlying_context)} · OI ${obs("InstrumentPriceDetails.OpenInterest", m.underlying_context)}`);
  text(`option-cost-${m.market}`, `Minimum purchase ${numeric(cost.minimum_purchase_cost_gbp, " GBP")} · entry fees ${numeric(cost.entry_costs_gbp)} · estimated exit ${numeric(cost.estimated_exit_costs_gbp)} · budget total ${numeric(cost.total_gbp)} / ${money(limits.per_trade_gbp)} · remaining ${numeric(cost.remaining_budget_gbp)} · ${cost.reason || cost.budget_result || "UNVERIFIED"}`);
  text(`option-coverage-${m.market}`, `Actual current option buffer ${Math.floor(o?.coverage_seconds || 0)} / 900s · subscription started ${o?.subscription_started_at ? new Date(o.subscription_started_at*1000).toISOString() : "unknown"} · latest event ${event?.id || "none"}: UIC ${event?.context?.identity?.uic || "unavailable"}, actual pre-trigger ${event ? Math.floor(event.context.pre_trigger_seconds || 0) + " / 900s" : "unavailable"}`);
  text(`option-meta-${m.market}`, $(`option-meta-${m.market}`).closest("details").open ? JSON.stringify(context, null, 2) : "");
  text(`option-quote-${m.market}`, $(`option-quote-${m.market}`).textContent + ` · bid size ${o?.size_status?.Bid || "UNVERIFIED"} · ask size ${o?.size_status?.Ask || "UNVERIFIED"}`);
}
function bookFlow(m, f, rec = {}, identity = {}) {
  if (route !== "markets") return;
  flowExpiry.set(m, (f.valid_until || 0) * 1000);
  text(`flow-status-${m}`, `${f.status || "UNAVAILABLE"} · ${(f.quality_flags || []).join(" · ")} · ${rec.state || "UNAVAILABLE"} · ${Math.floor(rec.prehistory_seconds || 0)} / 900s buffer`);
  const feed = f.feed || {}, cadence = feed.observed_receipt_ms || {};
  text(`flow-feed-${m}`, `${identity.environment || "UNVERIFIED"} · UIC ${identity.uic ?? "UNVERIFIED"} · delay ${numeric(f.delay_minutes, " min")} · granted ${numeric(feed.granted_refresh_ms, " ms")} · observed receipt mean ${numeric(cadence.mean, " ms")} (${cadence.samples || 0} intervals)`);
  text(`flow-spread-${m}`, `Spread ${numeric(f.spread_ticks, " ticks")} · usable depth up to 10: ${f.available_levels?.bid ?? 0} bid / ${f.available_levels?.ask ?? 0} ask levels`);
  for (const n of [1,3,5,10]) {
    const d = f.depth?.[n] || {}, row = $(`flow-depth-${m}-${n}`);
    [numeric(d.bid),numeric(d.ask),numeric(d.imbalance),numeric(d.order_imbalance),d.label || "UNAVAILABLE"].forEach((v,i) => { if(row.children[i+1].textContent !== v) row.children[i+1].textContent = v; });
  }
  text(`flow-mid-${m}`, `Size-weighted midpoint ${numeric(f.weighted_midpoint)} · displacement ${numeric(f.weighted_displacement_ticks, " ticks")}`);
  text(`flow-changes-${m}`, "Observed five-level depth changes · " + [5,30,60].map(s => {
    const d = f.lookbacks?.[s]?.[5];
    return `${s}s: ${d?.status === "AVAILABLE" ? `bid ${numeric(d.bid_change)} / ask ${numeric(d.ask_change)}` : "INSUFFICIENT_HISTORY"}`;
  }).join(" · "));
  const p = f.lookbacks?.[60]?.[5] || {};
  text(`flow-persistence-${m}`, `Five-level persistence over 60s · ${p.bid_heavy_fraction == null ? "INSUFFICIENT_HISTORY" : `BID_HEAVY ${numeric(p.bid_heavy_fraction * 100, "%")} / ASK_HEAVY ${numeric(p.ask_heavy_fraction * 100, "%")} / BALANCED ${numeric(p.balanced_fraction * 100, "%")}`}`);
  text(`flow-last-${m}`, `Latest-trade observation: ${numeric(f.latest_trade?.price)} × ${numeric(f.latest_trade?.size)} · no inferred execution count`);
  text(`flow-volume-${m}`, `Reported volume ${numeric(f.volume?.value)} · change UNAVAILABLE · ${f.volume?.status || "UNAVAILABLE"}`);
  if ($(`flow-detail-${m}`).open) text(`flow-meta-${m}`, JSON.stringify({version:f.version, calculated_at:f.at, last_receipt:feed.last_receipt, last_field_change:feed.last_field_change, last_contact:feed.last_contact, cadence, available_fields:f.available_fields, recording:rec, matched_price_changes:f.lookbacks}, null, 2));
}
function chart(m) {
  if (route !== "markets") return;
  const bars = m.chart || [],
    line = $(`line-${m.market}`),
    group = $(`markers-${m.market}`);
  text(`chart-empty-${m.market}`, bars.length ? "" : "Awaiting completed bars");
  if (!bars.length) {
    line.setAttribute("points", "");
    return;
  }
  const signature = JSON.stringify(bars);
  if (chartGeometry.get(m.market) !== signature) {
  chartGeometry.set(m.market, signature);
  const prices = bars.map((b) => b.close),
    lo = Math.min(...prices),
    span = Math.max(...prices) - lo || 1;
  const points = prices
    .map(
      (p, i) =>
        `${(i * 420) / Math.max(1, prices.length - 1)},${82 - ((p - lo) * 72) / span}`,
    )
    .join(" ");
  if (line.getAttribute("points") !== points)
    line.setAttribute("points", points);
  }
  const start = Date.parse(bars[0].at),
    end = Date.parse(bars.at(-1).at),
    wanted = new Set();
  const markers = [
    ...(m.signals || []).map((s) => ({
      id: `signal-${s.id}`,
      at: s.signal_at,
      label: `Opportunity · ${display(s.decision)} · ${display(s.reason)}`,
      kind: "signal",
    })),
    ...(m.trades || [])
      .filter((t) => t.entry_at)
      .map((t) => ({
        id: `fill-${t.id}`,
        at: t.entry_at,
        label: `Paper fill · ${t.basis || "basis unverified"}`,
        kind: "fill",
      })),
  ];
  for (const t of markers) {
    const x = (420 * (Date.parse(t.at) - start)) / Math.max(1, end - start);
    if (x < 0 || x > 420) continue;
    wanted.add(t.id);
    let mark = [...group.children].find((n) => n.dataset.key === t.id);
    if (!mark) {
      mark = document.createElementNS("http://www.w3.org/2000/svg", "line");
      mark.dataset.key = t.id;
      mark.dataset.kind = t.kind;
      mark.setAttribute("y1", "4");
      mark.setAttribute("y2", "88");
      const title = document.createElementNS(
        "http://www.w3.org/2000/svg",
        "title",
      );
      title.textContent = `${t.label} ${time(t.at)}`;
      mark.append(title);
      group.append(mark);
    }
    mark.setAttribute("x1", x);
    mark.setAttribute("x2", x);
  }
  [...group.children]
    .filter((n) => !wanted.has(n.dataset.key))
    .forEach((n) => n.remove());
}
function render(d) {
  const s = d.system,
    p = d.pnl || {skip_reasons:{}};
  p.skip_reasons ||= {};
  d.markets ||= [];
  limits = s.limits || limits;
  if (d.account) renderAccount(d.account);
  paused = s.paused;
  text("mode-banner", `DATA ENVIRONMENT: ${s.data_environment || "UNVERIFIED"} · EXECUTION MODE: ${s.execution_mode || "DISABLED"} · LIVE ORDERS DISABLED`);
  text("provider-status", "IBKR PARKED · FMP INACTIVE · EODHD INACTIVE");
  const exceptions = Object.values(s.management_problems || {}).join(" · ");
  text("execution-warning", exceptions || "No reported exposure exceptions");
  text("global-warning", exceptions || s.problem || s.l2_recording?.paused_reason || "");
  text("auth-state", `OAuth: ${s.oauth || "UNVERIFIED"}${s.oauth_problem ? ` (${s.oauth_problem})` : ""} · stream: ${s.connected ? "connected" : "disconnected"} · session: ${s.session?.TradeLevel || "UNVERIFIED"}`);
  for (let i = 0; i < 4; i++) {
    const label = i < s.reserved_open_trades ? `Slot ${i + 1} · reserved / open` : `Slot ${i + 1} · available`;
    text(`trade-slot-${i}`, label); text(`overview-slot-${i}`, label);
  }
  text(
    "connection",
    s.connected
      ? s.reconciled
        ? "Connected · reconciled"
        : "Connected · reconciliation required"
      : "Disconnected",
  );
  text("capacity", `${s.reserved_open_trades} / ${limits.slots ?? "—"}`);
  text("allocation", `${money(s.allocation_pennies / 100)} / ${money(limits.allocation_gbp)}`);
  if (d.pnl) {
  text("realised", money(p.realised_net_gbp));
  text(
    "pnl-note",
    p.provisional_closed
      ? `${p.provisional_closed} closed trade(s) provisional · missing costs/FX`
      : display(p.basis || "Paper evidence unavailable"),
  );
  }
  text(
    "entries",
    s.paused ? "Paused" : s.armed ? "Armed · readiness gates apply" : "Unarmed",
  );
  text("pause", paused ? "Resume entries" : "Pause entries");
  text("readiness", s.entry_block_reason ? display(s.entry_block_reason) : s.paused ? "New entries paused" : "Execution gates ready · market checks still apply");
  $("readiness").title = s.entry_block_reason || "";
  if (d.pnl) {
  const vals = d.markets
    .flatMap((m) => m.trades)
    .filter((t) => t.quantity > 0)
    .map((t) => t.valuation);
  text(
    "unrealised",
    vals.length && vals.every((v) => v.fresh && v.value_gbp != null)
      ? money(vals.reduce((a, v) => a + v.value_gbp, 0))
      : "Unavailable",
  );
  text(
    "valuation-note",
    vals.length
      ? "Bid estimate before commissions · quote/FX times in details"
      : "No open paper positions",
  );
  }
  for (const m of d.markets) {
    if (!markets.includes(m.market)) continue;
    if (route !== "markets") {
      text(`state-${m.market}`, display(m.strategy_state));
      text(`contract-${m.market}`, m.contract || "Contract awaiting verification");
      text(`block-${m.market}`, m.pending_data ? "Waiting for the completed boundary bar · original entry deadline applies" : display(m.block_reason));
      $(`block-${m.market}`).title = m.block_reason || "";
      text(`position-${m.market}`, m.trades?.length ? m.trades.map(t => `${display(t.state)} · ${t.quantity} contract · exit ${time(t.exit_at)}`).join(" · ") : "No pending or open paper trade");
      text(`candidate-${m.market}`, m.candidate_uic ? `Candidate UIC ${m.candidate_uic}` : "No option candidate available");
      text(`next-${m.market}`, time(m.next_time, "America/New_York"));
      text(`data-${m.market}`, `${display(m.data_status)} · last quote ${m.last_receipt ? time(new Date(m.last_receipt*1000).toISOString()) : "Unavailable"}`);
      continue;
    }
    quoteReceipts.set(m.market, Date.parse(m.l1?.last_receipt));
    text(`capability-${m.market}`, `${m.market} · ${m.l1?.status || "UNVERIFIED"} · ${m.l2?.status || "L2_UNAVAILABLE"} · ${m.block_reason || "monitoring"}`);
    const q = m.l1?.quote || {}, sizes = m.l1?.sizes || {}, rec = m.recorder || {};
    text(`quote-${m.market}`, `Bid ${q.Bid ?? "—"} × ${sizes.bid ?? "—"} · Ask ${q.Ask ?? "—"} × ${sizes.ask ?? "—"} · spread ${m.l1?.spread ?? "—"} · delay ${m.l1?.delay_minutes ?? "unknown"} min`);
    text(`recorder-${m.market}`, `${rec.state || "UNAVAILABLE"} · ${Math.floor(rec.prehistory_seconds || 0)} / 900s prehistory · ${display(rec.reason)}`);
    text(`contract-${m.market}`, m.contract || "Contract pending verification");
    text(`state-${m.market}`, display(m.strategy_state));
    text(`market-${m.market}`, display(m.market_status));
    text(
      `data-${m.market}`,
      m.updated_at
        ? `Bars ${display(m.data_status)} · ${Math.max(0, Math.floor((Date.now() - Date.parse(m.updated_at)) / 1000))}s`
        : `Feed ${display(m.data_status)}`,
    );
    text(`l1-${m.market}`, `L1 ${display(m.l1?.status || "DISCONNECTED")}`);
    depth(m.market, m.l2 || { status: "DISABLED" });
    bookFlow(m.market, m.book_flow || {}, rec, m.identity || {environment:s.data_environment});
    optionContext(m);
    text(`direction-${m.market}`, m.direction);
    text(
      `conditions-${m.market}`,
      m.conditions?.rv15 != null
        ? `RV15 ${(m.conditions.rv15 * 100).toFixed(4)}% · completed bars`
        : "Waiting for required history",
    );
    text(
      `block-${m.market}`,
      m.entry_enabled
        ? "Paper entry enabled at frozen clocks"
        : display(m.block_reason) || "Outside entry window",
    );
    text(
      `position-${m.market}`,
      m.trades.length
        ? m.trades
            .map(
              (t) =>
                `${display(t.state)} · ${t.quantity} contract · exit ${time(t.exit_at)}${t.quantity ? ` · bid P&L ${t.valuation.fresh ? money(t.valuation.value_gbp) : "unavailable"}` : ""}`,
            )
            .join(" | ")
        : "No pending or open paper trade",
    );
    text(`next-${m.market}`, time(m.next_time, "America/New_York"));
    text(`diagnostic-${m.market}`, m.diagnostic);
    if ($(`details-${m.market}`).closest("details").open) text(
      `details-${m.market}`,
      JSON.stringify(
        {
          ...m.details,
          l1: m.l1,
          l2: m.l2,
          recorder: m.recorder,
          capabilities: m.capabilities,
          exchange_trade_date: m.exchange_trade_date,
          next_market_time: m.next_market_time,
          trades: m.trades,
        },
        null,
        2,
      ),
    );
    chart(m);
    $(`card-${m.market}`).dataset.state = m.strategy_state;
  }
  if (d.pnl) text(
    "evidence",
    `${p.opportunities} opportunities · ${p.eligible_trades} eligible trades · ${p.skip_reasons.MINIMUM_CONTRACT_COST_EXCEEDS_BUDGET || 0} budget skips · ${p.skip_reasons.SKIP_CAPACITY_FULL || 0} capacity skips · ${p.fills} executions · ${p.win_rate == null ? "Win rate unavailable" : `${(p.win_rate * 100).toFixed(1)}% win rate`} (${p.wins}/${p.closed_with_complete_costs} closed with complete costs)`,
  );
  const a = s.market_data || {}, l = s.l2_recording || {};
  text("api-lines", `${a.owned_lines ?? "—"} / ${a.app_budget ?? 32} subscriptions`);
  text("api-allowance", `Saxo session: ${s.session?.TradeLevel || "UNVERIFIED"}`);
  text("api-external", "Session upgrade is explicit; it can downgrade another Saxo application");
  text("api-depth", `${d.markets.filter(m => m.l2?.status === "L2_AVAILABLE").length} / 5 markets with received depth`);
  text("api-assignments", "CL · GC · NG · NQ · SI · independent server subscriptions");
  text("api-options", `${a.option_lines ?? 0} / ${a.option_budget || 16} regular option quote subscriptions`);
  text("api-pacing", JSON.stringify(a.rate_limits || {}));
  text("api-queue", `${a.rest_queue ?? 0} / 32 queued REST requests`);
  text(
    "api-storage",
    `${((l.disk_bytes || 0) / 1048576).toFixed(1)} / ${((l.disk_limit || 0) / 1048576).toFixed(0)} MiB stored`,
  );
  text(
    "api-gaps",
    `${l.recording_gaps ?? 0} recording gaps · ${l.writer_queue ?? 0} queued batches`,
  );
  text(
    "api-error",
    l.paused_reason ||
      a.errors?.at(-1)?.message ||
      "No reported entitlement or capacity errors",
  );
}
function depth(m, d) {
  if (route !== "markets") return;
  const expires = d.valid_until != null ? d.valid_until * 1000 : Date.parse(d.last_receipt) + 5000;
  depthReceipts.set(m, d.fresh ? expires : 0);
  const covered = Math.floor(d.pre_seconds || 0);
  text(
    `l2-${m}`,
    `L2 ${display(d.status)}${d.target_pre_seconds ? ` · ${covered} / ${d.target_pre_seconds}s pre-context` : ""}`,
  );
  const fresh = d.fresh && Date.now() <= expires;
  text(
    `depth-note-${m}`,
    `${display(d.reason)}${fresh ? ` · ${d.bids?.length || 0} bid / ${d.asks?.length || 0} ask levels received · Last depth receipt ${time(d.last_receipt)}` : " · No current ladder"}`,
  );
  for (let i = 0; i < 20; i++) {
    const row = $(`depth-${m}-${i}`),
      bid = fresh ? d.bids?.[i] : null,
      ask = fresh ? d.asks?.[i] : null;
    row.hidden = !bid && !ask;
    [bid?.size, bid?.price, ask?.price, ask?.size].forEach((v, j) => {
      const value = String(v ?? "—");
      if (row.children[j].textContent !== value)
        row.children[j].textContent = value;
    });
  }
}
async function request(path, {method = "GET", signal, timeout = 8000} = {}) {
  const controller = new AbortController();
  const cancel = () => controller.abort();
  signal?.addEventListener("abort", cancel, {once:true});
  if (signal?.aborted) controller.abort();
  const timer = setTimeout(() => controller.abort("timeout"), timeout);
  try {
    const response = await fetch(path, {method, cache:"no-store", credentials:"same-origin",
      headers:{Accept:"application/json"}, signal:controller.signal});
    const result = await response.json().catch(error => {if(controller.signal.aborted) throw error; return null;});
    if (!response.ok) throw Error(result?.detail || result?.error || `Request failed (${response.status})`);
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
  const native = v => typeof v === "number" && a.currency ? `${new Intl.NumberFormat("en-GB", {maximumFractionDigits:2,minimumFractionDigits:2}).format(v)} ${a.currency}` : "Unavailable";
  text("account-value", native(a.total_value));
  text("account-cash", native(a.cash_balance));
  text("account-available", native(a.cash_available_for_trading));
  text("account-freshness", `${a.status} · last successful update ${a.last_success_at != null ? time(new Date(a.last_success_at*1000).toISOString()) : "Unavailable"}`);
  if ($("account-detail").open) text("account-json", JSON.stringify({basis:a.basis, ...a.details, problem:a.problem}, null, 2));
}
function rows(id, items, columns, key) {
  const body = $(id), known = new Map([...body.children].map(n => [n.dataset.key,n]));
  for (const [index,item] of items.entries()) {
    const identity = String(item[key]);
    let node = known.get(identity); known.delete(identity);
    if (!node) {node = document.createElement("tr"); node.dataset.key=identity; columns.forEach(() => node.append(document.createElement("td")));}
    columns.forEach((f,i) => { const next=String(f(item) ?? "Unavailable"); if(node.children[i].textContent!==next) node.children[i].textContent=next; });
    if (body.children[index] !== node) body.insertBefore(node, body.children[index] || null);
  }
  known.forEach(n=>n.remove());
}
function renderExecution(d) {
  render({system:d.system});
  rows("execution-trades", d.trades, [t=>t.market,t=>display(t.state),t=>t.option.symbol || t.option.uic,t=>t.quantity,t=>money(t.allocation_pennies/100),t=>time(t.exit_at)], "id");
  rows("execution-orders", d.orders, [o=>o.role,o=>o.order_id,o=>display(o.status),o=>o.filled,o=>o.remaining,o=>time(o.deadline)], "reference");
  rows("execution-fills", d.fills, [f=>time(f.at),f=>f.con_id,f=>f.side,f=>f.quantity,f=>f.price,f=>f.commission == null ? "Unavailable" : `${f.commission} ${f.commission_currency || "currency unverified"}`], "exec_id");
  rows("execution-positions", d.positions, [p=>p.con_id,p=>p.quantity], "con_id");
  text("execution-empty", d.trades.length ? "Internal reservations include pending entries and open trades." : "No pending or open SLRNO trades.");
  if ($("execution-detail").open) text("execution-json", JSON.stringify(d,null,2));
}
async function history(signal) {
  const q = new URLSearchParams({ offset: String(offset) });
  for (const [key, id] of [
    ["market", "market-filter"],
    ["day", "day-filter"],
    ["version", "version-filter"],
  ])
    if ($(id).value) q.set(key, $(id).value);
  q.set("sort", $("sort-filter").value);
  const d = await request(`/api/history?${q}`, {signal}),
    rows = [...d.rows];
  if (signal.aborted) return;
  if (d.system) render({system:d.system});
  const body = $("history"),
    known = new Map([...body.children].map((n) => [n.dataset.key, n]));
  for (const [i, r] of rows.entries()) {
    let tr = known.get(r.id);
    known.delete(r.id);
    if (!tr) {
      tr = document.createElement("tr");
      tr.dataset.key = r.id;
      for (let j = 0; j < 6; j++) tr.append(document.createElement("td"));
      const b = document.createElement("button");
      b.textContent = "Inspect";
      b.onclick = () => showDetail(r.id).catch(reportError);
      tr.children[5].append(b);
    }
    [
      time(r.signal_at),
      r.market,
      display(r.state || r.decision),
      display(r.reason) || "—",
      r.rule_version,
    ].forEach((v, j) => {
      if (tr.children[j].textContent !== v) tr.children[j].textContent = v;
    });
    if (body.children[i] !== tr)
      body.insertBefore(tr, body.children[i] || null);
  }
  known.forEach((n) => n.remove());
  $("history-empty").hidden = rows.length > 0;
  $("prev").disabled = offset === 0;
  $("next").disabled = !d.has_more;
  text("page-number", `Page ${1 + offset / 100}`);
  if (selectedIdentity && $("trade-detail").open) await showDetail(selectedIdentity, false);
}
async function showDetail(id, expand = true) {
  selectedIdentity = id;
  detailRequest?.abort();
  const controller = new AbortController(); detailRequest=controller;
  if (expand) $("trade-detail").open = true;
  const d = await request(`/api/detail?identity=${encodeURIComponent(id)}`, {signal:controller.signal});
  if (!controller.signal.aborted && selectedIdentity === id) text("trade-json", JSON.stringify(d,null,2));
}
function reportError(e) {
  if (e.name !== "AbortError") text("notice", `${e.message} · displayed values are stale`);
}
async function refresh(force = false) {
  if (document.hidden || controlPending || (pending && !force)) return;
  if (force) pending?.abort();
  const controller = new AbortController(); pending=controller;
  try {
    if (page === "trades") await history(controller.signal);
    else {
      let path = "/api/overview";
      if (route === "markets") path = `/api/market/${selectedMarket}?diagnostics=${!!$(`card-${selectedMarket}`).querySelector("details[open]")}`;
      if (page === "execution") path = "/api/execution";
      if (page === "system") path = `/api/system?diagnostics=${$("system-detail").open}`;
      const d = await request(path, {signal:controller.signal});
      if (controller.signal.aborted) return;
      if (page === "execution") renderExecution(d);
      else if (page === "system") {
        render({system:d});
        text("api-depth", `${(d.markets || []).filter(m=>m.capabilities?.l2?.status === "L2_AVAILABLE" || m.l2?.status === "L2_AVAILABLE").length} / 5 markets with received depth`);
        for (const m of d.markets || []) text(`capability-${m.market}`, `${m.market} · ${display(m.problem) || "Connected"}`);
        if ($("system-detail").open) text("system-json",JSON.stringify(d,null,2));
        if ($("recordings-detail").open) {
          const recordings = await request("/api/recordings", {signal:controller.signal});
          if (controller.signal.aborted) return;
          const entries = [...(recordings.active || []), ...(recordings.completed || [])].slice(-132);
          const wanted = new Set(entries.map(r => `recording-${r.segment}`));
          for(const node of [...$("recording-list").children]) if(!wanted.has(node.id)) node.remove();
          for(const rec of entries) {
            const id=`recording-${rec.segment}`;
            if(!$(id)) {const a=document.createElement("a"); a.id=id; a.href=`/api/recordings/${encodeURIComponent(rec.segment)}`; $("recording-list").append(a);}
            text(id,`${rec.state} · ${rec.key} · ${rec.segment.slice(0,10)}`);
          }
        }
      } else render(d);
    }
    if (!controller.signal.aborted) {
      lastRefresh = new Date().toISOString(); text("notice", "");
      text("last-refresh", `Last successful refresh ${time(lastRefresh)}`);
    }
  } catch(e) {reportError(e);} finally {if(pending===controller) pending=null;}
}
$("saxo-connect").onsubmit = async event => {
  event.preventDefault();
  const button=event.currentTarget.querySelector("button"); button.disabled=true;
  text("oauth-notice", "");
  try {const result=await request("/oauth/saxo/start", {method:"POST",timeout:15000}); window.location.assign(result.authorization_url);}
  catch(e) {text("oauth-notice",e.message); button.disabled=false;}
};
$("pause").onclick = async () => {
  pending?.abort(); controlPending=true; $("pause").disabled=true;
  try {const result=await request(`/api/entries/${paused ? "resume" : "pause"}`, {method:"POST"}); render({system:result}); text("notice",result.paused ? "Server confirmed: entries paused" : "Server confirmed: entries resumed; readiness gates still apply");}
  catch(e) {reportError(e);} finally {controlPending=false; $("pause").disabled=false;}
};
for(const id of ["market-filter","day-filter","version-filter","sort-filter"])
  $(id).addEventListener("change",()=>{offset=0; selectedIdentity=""; detailRequest?.abort(); text("trade-json", "Select an opportunity to view its evidence"); refresh(true);});
$("prev").onclick=()=>{offset=Math.max(0,offset-100);refresh(true);};
$("next").onclick=()=>{offset+=100;refresh(true);};
document.addEventListener("visibilitychange",()=>{if(document.hidden){pending?.abort(); detailRequest?.abort();}else refresh(true);});
document.querySelectorAll("details").forEach(el=>el.addEventListener("toggle",()=>{
  if (el.id === "account-detail") {if(accountSnapshot) renderAccount(accountSnapshot);}
  else if (el.open) refresh(true);
}));
setInterval(() => {
  text("clock", `${time(new Date().toISOString())} · London`);
  if (accountSnapshot?.status === "Current" && Date.now() > accountSnapshot.valid_until*1000) renderAccount({...accountSnapshot,status:"Stale"});
  if (lastRefresh && Date.now()-Date.parse(lastRefresh)>15000) text("last-refresh",`Stale · last successful refresh ${time(lastRefresh)}`);
  for (const [m, at] of depthReceipts)
    if (at && Date.now() > at)
      depth(m, { status: "L2_UNAVAILABLE", reason: "SUBSCRIPTION_HEALTH_EXPIRED" });
  for (const [m, expires] of flowExpiry) if (expires && Date.now() > expires) bookFlow(m, {status:"UNAVAILABLE",quality_flags:["SUBSCRIPTION_HEALTH_EXPIRED"]});
  for (const [m, at] of quoteReceipts) if (!at || Date.now() - at > 5000) text(`l1-${m}`, "L1 STALE OR MISSING");
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
$("market-selector").hidden = route !== "markets";
if (route === "markets") $("markets").classList.add("single-market");
const linkedMarket = new URLSearchParams(location.search).get("market");
if (markets.includes(linkedMarket)) selectedMarket = linkedMarket;
$("selected-market").value = selectedMarket;
$("account-strip").hidden = route !== "overview";
$("strategy-summary").hidden = route !== "overview";
$("overview-slots").hidden = route !== "overview";
function selectMarket() {
  selectedMarket = $("selected-market").value;
  sessionStorage.setItem("slrno-market", selectedMarket);
  for (const market of markets) $(`card-${market}`).hidden = route === "markets" && market !== selectedMarket;
  refresh(true);
}
$("selected-market").addEventListener("change", selectMarket);
selectMarket();
if (route === "markets") text("title", "Markets");
if (route === "opportunities") text("title", "Opportunities");
if (route === "execution") text("title", "Execution");
