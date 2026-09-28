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
  text("execution-warning", Object.values(s.management_problems || {}).join(" · ") || "No reported exposure exceptions");
  for (let i = 0; i < 4; i++) text(`trade-slot-${i}`, i < s.reserved_open_trades ? `Slot ${i + 1} · reserved / open` : `Slot ${i + 1} · available`);
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
  depthReceipts.set(m, d.fresh ? Date.parse(d.last_receipt) : 0);
  const covered = Math.floor(d.pre_seconds || 0);
  text(
    `l2-${m}`,
    `L2 ${display(d.status)}${d.target_pre_seconds ? ` · ${covered} / ${d.target_pre_seconds}s pre-context` : ""}`,
  );
  const fresh = d.fresh && Date.now() - Date.parse(d.last_receipt) <= 5000;
  text(
    `depth-note-${m}`,
    `${display(d.reason)}${fresh ? ` · Local receipt ${time(d.last_receipt)}` : " · No current ladder"}`,
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
    if (at && Date.now() - at > 5000)
      depth(m, { status: "INCOMPLETE", reason: "LADDER_RECEIPT_STALE" });
}, 1000);
setInterval(refresh, 5000);
refresh();

for (let i = 0; i < 4; i++) {
  const slot = document.createElement("div"); slot.id = `trade-slot-${i}`;
  $("trade-slots").append(slot);
}
$("market-selector").hidden = route !== "markets";
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
