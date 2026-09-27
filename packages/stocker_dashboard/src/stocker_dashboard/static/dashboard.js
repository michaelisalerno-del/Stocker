"use strict";
const $ = (id) => document.getElementById(id);
const markets = ["BTC", "CL", "GC", "NG", "NQ", "SI"];
const names = {
  BTC: "Bitcoin",
  CL: "Crude oil",
  GC: "Gold",
  NG: "Natural gas",
  NQ: "Nasdaq 100",
  SI: "Silver",
};
const page =
  location.pathname === "/trades"
    ? "trades"
    : location.pathname === "/system"
      ? "system"
      : "overview";
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
for (const name of ["overview", "trades", "system"])
  $(name).hidden = name !== page;
text(
  "title",
  page === "trades"
    ? "Trades & signal history"
    : page === "system"
      ? "System"
      : "Futures overview",
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
 <svg class="chart" viewBox="0 0 420 92" role="img" aria-label="${m} completed underlying prices"><path class="chart-grid" d="M0 30H420 M0 62H420"/><polyline id="line-${m}" fill="none"/><g id="markers-${m}"></g><text id="chart-empty-${m}" x="210" y="48" text-anchor="middle">Awaiting completed bars</text></svg>
 <div class="condition"><span id="direction-${m}"></span><strong id="conditions-${m}"></strong></div>
 <p class="block" id="block-${m}"></p><div class="position" id="position-${m}"></div>
 <div class="next"><span>Next clock · NY</span><strong id="next-${m}"></strong></div>
 <p class="diagnostic" id="diagnostic-${m}"></p><details><summary>Contract & rule details</summary><pre id="details-${m}"></pre></details>`;
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
        label: "Broker paper fill",
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
      : "Broker executions + commissions",
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
    text(`contract-${m.market}`, m.contract || "Contract pending verification");
    text(`state-${m.market}`, display(m.strategy_state));
    text(`market-${m.market}`, display(m.market_status));
    text(
      `data-${m.market}`,
      m.updated_at
        ? `${display(m.data_status)} · ${Math.max(0, Math.floor((Date.now() - Date.parse(m.updated_at)) / 1000))}s`
        : display(m.data_status),
    );
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
    `${p.opportunities} opportunities · ${p.eligible_trades} eligible trades · ${p.skip_reasons.SKIP_BUDGET_TOO_SMALL || 0} budget skips · ${p.skip_reasons.SKIP_CAPACITY_FULL || 0} capacity skips · ${p.fills} executions · ${p.win_rate == null ? "Win rate unavailable" : `${(p.win_rate * 100).toFixed(1)}% win rate`} (${p.wins}/${p.closed_with_complete_costs} closed with complete costs)`,
  );
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
    text("notice", "");
  } catch (e) {
    text("notice", `${e.message} · last displayed values may be stale`);
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
setInterval(
  () => text("clock", `${time(new Date().toISOString())} · London`),
  1000,
);
setInterval(refresh, 5000);
refresh();
