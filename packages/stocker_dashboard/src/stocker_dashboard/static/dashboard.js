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
  pending = false,
  selectedIdentity = "";
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
const display = (v) => String(v || "").replaceAll("_", " ");
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
  card.innerHTML = `<div class="card-head"><div><h2>${m} <span>${names[m]}</span></h2><p id="contract-${m}"></p></div><span class="state" id="state-${m}"></span></div>
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
    card.insertBefore(panel, card.querySelector("details"));
    panel.append(card.querySelector(".depth"));
  }
}
const numeric = (v, suffix = "") => typeof v === "number" && Number.isFinite(v) ? `${Number(v.toFixed(4))}${suffix}` : "UNAVAILABLE";
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
  text(`flow-meta-${m}`, JSON.stringify({version:f.version, calculated_at:f.at, last_receipt:feed.last_receipt, last_field_change:feed.last_field_change, last_contact:feed.last_contact, cadence, available_fields:f.available_fields, recording:rec, matched_price_changes:f.lookbacks}, null, 2));
}
function chart(m) {
  const bars = m.chart || [],
    line = $(`line-${m.market}`),
    group = $(`markers-${m.market}`);
  text(`chart-empty-${m.market}`, bars.length ? "" : "Awaiting completed bars");
  if (!bars.length) {
    line.setAttribute("points", "");
    return;
  }
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
    p = d.pnl;
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
  text("account", s.account);
  text(
    "connection",
    s.connected
      ? s.reconciled
        ? "Connected · reconciled"
        : "Connected · reconciliation required"
      : "Disconnected",
  );
  text("capacity", `${s.reserved_open_trades} / 4`);
  text("allocation", `${money(s.allocation_pennies / 100)} / £40`);
  text("realised", money(p.realised_net_gbp));
  text(
    "pnl-note",
    p.provisional_closed
      ? `${p.provisional_closed} closed trade(s) provisional · missing costs/FX`
      : display(p.basis || "Paper evidence unavailable"),
  );
  text(
    "entries",
    s.paused ? "Paused" : s.armed ? "Armed · readiness gates apply" : "Unarmed",
  );
  text("pause", paused ? "Resume entries" : "Pause entries");
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
  for (const m of d.markets) {
    if (!markets.includes(m.market)) continue;
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
    text(`direction-${m.market}`, m.direction);
    text(
      `conditions-${m.market}`,
      m.conditions.rv15 != null
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
    text(
      `details-${m.market}`,
      JSON.stringify(
        {
          ...m.details,
          l1: m.l1,
          l2: m.l2,
          recorder: m.recorder,
          options: m.option,
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
  text(
    "evidence",
    `${p.opportunities} opportunities · ${p.eligible_trades} eligible trades · ${p.skip_reasons.MINIMUM_CONTRACT_COST_EXCEEDS_BUDGET || 0} budget skips · ${p.skip_reasons.SKIP_CAPACITY_FULL || 0} capacity skips · ${p.fills} executions · ${p.win_rate == null ? "Win rate unavailable" : `${(p.win_rate * 100).toFixed(1)}% win rate`} (${p.wins}/${p.closed_with_complete_costs} closed with complete costs)`,
  );
  const a = s.market_data || {}, l = s.l2_recording || {};
  text("api-lines", `${a.owned_lines ?? "—"} / ${a.app_budget ?? 32} subscriptions`);
  text("api-allowance", `Saxo session: ${s.session?.TradeLevel || "UNVERIFIED"}`);
  text("api-external", "Session upgrade is explicit; it can downgrade another Saxo application");
  text("api-depth", `${d.markets.filter(m => m.l2?.status === "L2_AVAILABLE").length} / 5 markets with received depth`);
  text("api-assignments", "CL · GC · NG · NQ · SI · independent server subscriptions");
  text("api-options", `${d.markets.reduce((n,m) => n + (m.option?.quotes?.length || 0),0)} / 16 regular option quote subscriptions`);
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
async function get(path) {
  const r = await fetch(path, { cache: "no-store" });
  if (!r.ok) throw Error(`Data unavailable (${r.status})`);
  return r.json();
}
async function history() {
  const q = new URLSearchParams({ offset: String(offset) });
  for (const [key, id] of [
    ["market", "market-filter"],
    ["day", "day-filter"],
    ["version", "version-filter"],
  ])
    if ($(id).value) q.set(key, $(id).value);
  const d = await get(`/api/history?${q}`),
    rows = [...d.rows];
  if ($("sort-filter").value === "asc") rows.reverse();
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
      b.onclick = () => showDetail(r.id);
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
  if (selectedIdentity) await showDetail(selectedIdentity, false);
}
async function showDetail(id, expand = true) {
  selectedIdentity = id;
  text(
    "trade-json",
    JSON.stringify(
      await get(`/api/detail?identity=${encodeURIComponent(id)}`),
      null,
      2,
    ),
  );
  if (expand) $("trade-detail").open = true;
}
async function refresh() {
  if (pending || document.hidden) return;
  pending = true;
  try {
    render(await get("/api/overview"));
    if (page === "trades") await history();
    if (page === "system")
      text("system-json", JSON.stringify(await get("/api/system"), null, 2));
    if (page === "system") {
      const recordings = await get("/api/recordings");
      const entries = [...(recordings.active || []), ...(recordings.completed || [])].slice(-132);
      const wanted = new Set(entries.map(r => `recording-${r.segment}`));
      for (const node of [...$("recording-list").children]) if (!wanted.has(node.id)) node.remove();
      for (const rec of entries) {
        const id = `recording-${rec.segment}`;
        if (!$(id)) {
          const link = document.createElement("a"); link.id = id;
          link.href = `/api/recordings/${encodeURIComponent(rec.segment)}`;
          $("recording-list").append(link);
        }
        text(id, `${rec.state} · ${rec.key} · ${rec.segment.slice(0, 10)}`);
      }
    }
    if (page === "execution") text("execution-json", JSON.stringify(await get("/api/history"), null, 2));
    text("notice", "");
  } catch (e) {
    text("notice", `${e.message} · last displayed values may be stale`);
    for (const m of markets)
      depth(m, { status: "UNAVAILABLE", reason: "DASHBOARD_DATA_STALE" });
  } finally {
    pending = false;
  }
}
$("saxo-connect").onsubmit = async (event) => {
  event.preventDefault();
  const button = event.currentTarget.querySelector("button");
  button.disabled = true;
  text("oauth-notice", "");
  try {
    // A native form POST with no-referrer sends Origin: null. Fetch's default
    // cors mode preserves the Origin for this same-origin authenticated POST.
    const response = await fetch("/oauth/saxo/start", {
      method: "POST",
      headers: { Accept: "application/json" },
      credentials: "same-origin",
      signal: AbortSignal.timeout(15000),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || "Saxo connection could not start");
    window.location.assign(result.authorization_url);
  } catch (error) {
    text("oauth-notice", error.message);
    button.disabled = false;
  }
};
$("pause").onclick = async () => {
  await fetch(`/api/entries/${paused ? "resume" : "pause"}`, {
    method: "POST",
  });
  await refresh();
};
for (const id of [
  "market-filter",
  "day-filter",
  "version-filter",
  "sort-filter",
])
  $(id).addEventListener("change", () => {
    offset = 0;
    history().catch((e) => text("notice", e.message));
  });
$("prev").onclick = () => {
  offset = Math.max(0, offset - 100);
  history();
};
$("next").onclick = () => {
  offset += 100;
  history();
};
setInterval(() => {
  text("clock", `${time(new Date().toISOString())} · London`);
  for (const [m, at] of depthReceipts)
    if (at && Date.now() > at)
      depth(m, { status: "L2_UNAVAILABLE", reason: "SUBSCRIPTION_HEALTH_EXPIRED" });
  for (const [m, expires] of flowExpiry) if (expires && Date.now() > expires) bookFlow(m, {status:"UNAVAILABLE",quality_flags:["SUBSCRIPTION_HEALTH_EXPIRED"]});
  for (const [m, at] of quoteReceipts) if (!at || Date.now() - at > 5000) text(`l1-${m}`, "L1 STALE OR MISSING");
}, 1000);
setInterval(refresh, 5000);
refresh();

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
$("selected-market").value = selectedMarket;
function selectMarket() {
  selectedMarket = $("selected-market").value;
  sessionStorage.setItem("slrno-market", selectedMarket);
  for (const market of markets) $(`card-${market}`).hidden = route === "markets" && market !== selectedMarket;
}
$("selected-market").addEventListener("change", selectMarket);
selectMarket();
if (route === "markets") text("title", "Markets");
if (route === "opportunities") text("title", "Opportunities");
if (route === "execution") text("title", "Execution");
