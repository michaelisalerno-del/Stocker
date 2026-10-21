"use strict";
// SLRNO paper record. Display only: the server never places an order.
const $ = (id) => document.getElementById(id);
const route = location.pathname.replace(/^\/+|\/+$/g, "") || "today";
const page = ["today", "record", "system"].includes(route) ? route : "today";
const REFRESH_SECONDS = page === "today" ? 5 : 30;
const EVENTS = {
  FIRST_CANDLE: "First candle · CFD trades",
  FIRST_CANDLE_QQQ_ENTRY: "First candle · QQQ option bought",
  FIRST_CANDLE_QQQ_EXIT: "First candle · QQQ option sold",
  CONNORS_QQQ: "Connors check · QQQ call",
  FIRST_CANDLE_QQQ_LATE: "QQQ option 15:55 mark (reported)",
  CONNORS_QQQ_EXIT: "Connors · QQQ call sold",
  CFD_EXIT: "CFD trades closed",
  VWAP_CROSS_EXIT: "VWAP-crossing trades closed",
  DIP_EXIT: "Dip rules · trades closed",
  DIP_ENTRY: "Dip rules · Nasdaq CFD bought",
  STOCK_DIPS: "Stock dips · after-close scan of Saxo's daily bars: fills, exits, orders for the next session",
};
const MARKETS = {NAS100: "Nasdaq CFD", US500: "S&P CFD"};
const METHODS = {"CFD NAS100": "Nasdaq CFD · first candle", "CFD US500": "S&P CFD · first candle", FIRST_CANDLE_QQQ: "QQQ option · first candle", CONNORS_QQQ: "QQQ call · Connors", "CROSS NAS100": "Nasdaq CFD · VWAP crossings", "DIP CONNORS": "Nasdaq CFD · Connors (days)", "DIP TURNAROUND": "Nasdaq CFD · Turnaround Tuesday", "DIP FEAR": "Nasdaq CFD · fear rule (3 down closes, 30-day IV ≥ 30%)", "DIP BOOK": "Nasdaq CFD · dip book, one at a time", "STOCK DIP": "US stocks · Connors pullback (8% limit, out above ConnorsRSI 70)"};
const STOCK_STATUS = {PENDING: "Resting for the next session", NOT_FILLED: "Not filled", FILLED: "Filled", CANCELLED: "Cancelled"};
const methodName = (m) => METHODS[m] || (m.startsWith("STOCK DIP ") ? `${METHODS["STOCK DIP"]} · ${m.slice(10)}` : m);
const RULES = {CONNORS: "Connors", TURNAROUND: "Turnaround Tuesday", FEAR: "Fear rule"};
const BOOKS = {PULLBACK: "Pullback cell · long signal, prior close below the 50-day", WEEKDAYS: "Tuesday to Friday", AGREE: "Nasdaq and S&P signals agree", "NO OVERSOLD SHORTS": "Shorts skipped after an oversold close (3+ down closes or RSI2 < 10)", "AFTER A +1% DAY": "After a +1% day", PLAYBOOK: "Playbook · the call in the pullback cell, the Nasdaq CFD otherwise, no Monday or oversold shorts, S&P in the cell only", "SIZED V2": "Sized · playbook units (cell 2, other longs 1, shorts 1 unless Monday or oversold)", "SIZED V3": "Sized · score units 0-3 (+cell, +after a +1% day, +agree, −Monday, −oversold short)", "STOP 75 BP": "Nasdaq CFD · what a 75 bp stop at the same $200 risk would have done", "CROSS PULLBACK": "VWAP crossings · pullback days", "CROSS BEFORE 14:00": "VWAP crossings · entered before 14:00"};
const USD = new Intl.NumberFormat("en-US", {style: "currency", currency: "USD"});
let offset = 0, pending = null, lastRefresh = "";

const text = (id, value) => { const el = $(id); if (el && el.textContent !== String(value)) el.textContent = String(value); };
const show = (id, on) => { const el = $(id); if (el) el.hidden = !on; };
const display = (code) => code ? String(code).replace(/_/g, " ").toLowerCase().replace(/^./, (c) => c.toUpperCase()) : "";
const clockAt = (iso, zone) => iso ? new Date(iso).toLocaleTimeString("en-GB", {hour: "2-digit", minute: "2-digit", timeZone: zone}) : "—";
const ny = (iso) => clockAt(iso, "America/New_York");
const uk = (iso) => clockAt(iso, "Europe/London");
const money = (v) => v == null ? "—" : (v > 0 ? "+" : "") + USD.format(v);
const tone = (v) => v == null ? "" : v > 0 ? "pos" : v < 0 ? "neg" : "";
const side = (s) => s > 0 ? "Long" : s < 0 ? "Short" : "None";
const price = (v) => v == null ? "—" : Number(v).toFixed(2);

function rows(id, items, columns, classes = {}) {
  const body = $(id);
  body.replaceChildren(...items.map((item) => {
    const tr = document.createElement("tr");
    const labels = [...body.closest("table").querySelectorAll("thead th")].map((th) => th.textContent);
    columns.forEach((f, i) => {
      const td = document.createElement("td");
      td.dataset.label = labels[i] || "";
      td.textContent = String(f(item) ?? "—");
      const cls = classes[i]?.(item);
      if (cls) td.className = cls;
      tr.append(td);
    });
    return tr;
  }));
  show(`${id}-empty`, items.length === 0);
  body.closest(".table-scroll").hidden = items.length === 0;
}

async function request(path, {method = "GET", signal, timeout = 8000} = {}) {
  const controller = new AbortController();
  const cancel = () => controller.abort();
  signal?.addEventListener("abort", cancel, {once: true});
  const timer = setTimeout(() => controller.abort("timeout"), timeout);
  try {
    const response = await fetch(path, {method, cache: "no-store", credentials: "same-origin", referrerPolicy: "no-referrer", headers: {Accept: "application/json"}, signal: controller.signal});
    const result = await response.json().catch(() => null);
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

function renderStatus(s) {
  const login = s.oauth === "AUTHENTICATED";
  text("login-chip", login ? `Logged in · ${s.login_minutes_left ?? "?"} min left` : `Login: ${display(s.oauth)}`);
  $("login-chip").className = login ? "chip" : "chip bad";
  text("day-chip", s.session ? `US session ${ny(s.session.open)}–${ny(s.session.close)} NY${s.session.regular ? "" : " · early close, skipped"}` : "No US session today");
  text("delay-chip", s.qqq_delay_minutes == null ? "QQQ delay not measured yet" : s.qqq_delay_minutes ? `QQQ prices ${s.qqq_delay_minutes} min late` : "QQQ prices live");
  const problems = Object.entries(s.problems || {});
  show("problem-chip", problems.length > 0);
  text("problem-chip", problems.map(([k, v]) => `${display(k)}: ${display(v)}`).join(" · "));
  $("problem-chip").className = "chip bad";
  $("status").dataset.tone = !login || s.worker_health !== "RUNNING" ? "bad" : problems.length ? "warn" : "ok";
}

function signalText(s) {
  if (!s) return "Not yet";
  const d = s.detail || {};
  if (s.method === "CONNORS") {
    if (s.reason) return display(s.reason);
    const sum = d.rsi2 != null && d.rsi2_prev != null ? (d.rsi2 + d.rsi2_prev).toFixed(1) : "?";
    return `${s.side ? "Signal: buy the call" : "No signal"} · RSI2 sum ${sum} · price ${price(d.price_15_50)} vs 200-day ${price(d.sma200)}`;
  }
  if (s.method === "DIP") {
    if (s.reason) return display(s.reason);
    const connors = d.connors_in ? "Connors: buy" : d.connors_exit ? "Connors: RSI2 above 65, sell" : "Connors: no signal";
    const change = d.change == null ? "?" : `${(d.change * 100).toFixed(2)}%`;
    const fear = d.fear ? "Fear rule: buy" : d.atm_vol_30d == null ? `Fear rule: no reading${d.atm_vol_reason ? ` (${display(d.atm_vol_reason)})` : ""}` : `Fear rule: no signal (30-day IV ${(d.atm_vol_30d * 100).toFixed(0)}%, ${d.down_closes ?? "?"} down closes)`;
    return `${connors} · ${d.turnaround ? "Turnaround Tuesday: buy" : "Turnaround Tuesday: no signal"} · QQQ ${price(d.price_15_50)} (${change} on the day) · ${fear}`;
  }
  if (!s.side) return `No trade · ${display(s.reason) || "close equals EMA"}`;
  const trend = d.below_50d === true ? " · prior close below the 50-day" : d.below_50d === false ? " · prior close above the 50-day" : d.context_reason ? ` · context not read (${display(d.context_reason)})` : "";
  return `${side(s.side)} · candle ${price(d.close)} vs EMA12 ${price(d.ema)}${trend}`;
}

function playbookText(p) {
  if (!p) return "Waiting for the 09:35 signal";
  return `${p.action.charAt(0).toUpperCase()}${p.action.slice(1)}${p.tags.length ? ` · ${p.tags.join(" · ")}` : ""}`;
}
const playbookTone = (p) => !p ? "" : /^(skip|no trade|context not read)/.test(p.action) ? "skip" : "go";
const countdown = (iso) => {
  const ms = Date.parse(iso) - Date.now();
  if (!(ms > 0)) return "due now";
  const h = Math.floor(ms / 3600000), m = Math.floor((ms % 3600000) / 60000);
  return `in ${h ? `${h}h ` : ""}${m}m`;
};

function renderTiles(d) {
  const s = d.system, status = new Map(d.events.map((e) => [e.name, e]));
  const closed = [...d.cfd, ...(d.crosses || []), ...(d.options || []).filter((o) => o.exit_day === d.day), ...(d.dips || []).filter((t) => t.exit_day === d.day), ...(d.stock_trades || []).filter((t) => t.exit_day === d.day)]
    .reduce((sum, t) => sum + (t.pnl_usd ?? 0), 0);
  const open = [...d.cfd, ...(d.crosses || [])].reduce((sum, t) => sum + (t.exit_at ? 0 : (t.open_pnl_usd ?? 0)), 0);
  text("tile-pnl", money(closed + open)); $("tile-pnl").className = `tile-value ${tone(closed + open)}`;
  text("tile-pnl-sub", `closed ${money(closed)} · open ${money(open)}`);
  const positions = [
    ...d.cfd.filter((t) => !t.exit_at).map((t) => `${MARKETS[t.market] || t.market} ${side(t.side).toLowerCase()}`),
    ...(d.crosses || []).filter((t) => t.entry_price != null && !t.exit_at).map(() => "Nasdaq CFD crossing"),
    ...(d.dips || []).filter((t) => !t.exit_at).map((t) => `${RULES[t.rule] || t.rule} dip`),
    ...(d.stock_trades || []).filter((t) => !t.exit_day).map((t) => `${t.code} dip`),
    ...(d.options || []).filter((o) => !o.exit_read).map((o) => `QQQ ${o.put_call.toLowerCase()} ${o.strike}`),
  ];
  text("tile-open", positions.length); text("tile-open-sub", positions.join(" · ") || "Flat");
  const next = d.plan.find((p) => !["DONE", "FAILED", "MISSED"].includes(status.get(p.event)?.status));
  text("tile-next", next ? `${next.new_york} NY` : d.plan.length ? "Done for today" : "No session");
  text("tile-next-sub", next ? `${EVENTS[next.event] || next.event} · ${countdown(next.due)}` : "Nothing left to record today");
  const late = s.qqq_delay_minutes;
  text("tile-feed", late == null ? "QQQ delay unknown" : late ? `QQQ ${late} min late` : "QQQ live");
  $("tile-feed").className = `tile-value ${late == null ? "" : late ? "warn" : "pos"}`;
  const problems = Object.keys(s.problems || {}).length;
  text("tile-feed-sub", `CFD quotes live · login ${s.login_minutes_left ?? "?"} min · ${problems ? `${problems} problem${problems > 1 ? "s" : ""}` : "no problems"}`);
}

// chart --------------------------------------------------------------------------------------
const CHART_MARKETS = {NAS100: "Nasdaq CFD", US500: "S&P CFD", QQQ: "QQQ"};
const CHART_PANELS = [{box: "chart", stat: "chart-stat", market: "QQQ"}, {box: "chart-US500", stat: "chart-stat-US500", market: "US500"}];
let charts = null, todayData = null;
const SVG = "http://www.w3.org/2000/svg";
const svgEl = (tag, attrs = {}, content) => { const e = document.createElementNS(SVG, tag); for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v); if (content != null) e.textContent = content; return e; };
const nyFormat = new Intl.DateTimeFormat("en-CA", {timeZone: "America/New_York", year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hourCycle: "h23"});
function nyParts(ms) {
  const p = {}; for (const {type, value} of nyFormat.formatToParts(new Date(ms))) p[type] = value;
  return {date: `${p.year}-${p.month}-${p.day}`, hm: `${p.hour}:${p.minute}`};
}
const ema = (values, alpha) => { let v = null; return values.map((x) => (v = v == null ? x : v + alpha * (x - v))); };
const slot5 = (iso) => Math.floor(Date.parse(iso) / 300000) * 300000;

function chartLevels(chartMarket) {
  const d = todayData; if (!d) return [];
  if (chartMarket === "QQQ") {
    return (d.options || []).filter((o) => o.method === "FIRST_CANDLE_QQQ").map((o) => ({price: o.strike, cls: "c-strike", label: `${o.put_call.toLowerCase()} ${o.strike}`}));
  }
  const t = d.cfd.find((x) => x.market === chartMarket); if (!t) return [];
  const out = [{price: t.entry_price, cls: "c-entry", label: `entry ${side(t.side).toLowerCase()} ${price(t.entry_price)}`}];
  if (t.exit_at && t.exit_price != null) out.push({price: t.exit_price, cls: "c-exit", label: `exit ${display(t.exit_reason)}`});
  else out.push({price: t.stop, cls: "c-stop", label: `stop ${price(t.stop)}`});
  return out;
}

function chartMarkers(chartMarket, bars) {
  const d = todayData; if (!d) return [];
  const at = (ms) => bars.findIndex((b) => b.ms === ms);
  const out = [];
  if (chartMarket === "QQQ") {
    const fc = d.cfd.find((x) => x.market === "NAS100"); // the first-candle side, shown on QQQ at its 09:35 candle
    if (fc) out.push({index: at(slot5(fc.entry_at)), up: fc.side > 0, cls: fc.side > 0 ? "c-mark-up" : "c-mark-dn"});
    for (const c of d.crosses || []) {
      if (c.entry_price != null) out.push({index: at(slot5(c.signal_candle)), up: true, cls: "c-mark-up"});
      if (c.exit_candle) out.push({index: at(slot5(c.exit_candle)), up: false, cls: "c-mark-x"});
    }
  } else {
    const t = d.cfd.find((x) => x.market === chartMarket);
    if (t) { out.push({index: at(slot5(t.entry_at)), up: t.side > 0, cls: t.side > 0 ? "c-mark-up" : "c-mark-dn"}); if (t.exit_at && t.exit_price != null) out.push({index: at(slot5(t.exit_at)), up: t.side < 0, cls: "c-mark-x"}); }
  }
  return out.filter((m) => m.index >= 0);
}

function drawCharts() {
  for (const p of CHART_PANELS) drawChart(p.market, p.box, p.stat);
}

function drawChart(chartMarket, boxId, statId) {
  const box = $(boxId); if (!box) return;
  const data = charts?.markets?.[chartMarket];
  const all = (data?.bars || []).map((b) => { const ms = Date.parse(b[0]); return {ms, o: b[1], h: b[2], l: b[3], c: b[4], v: b[5], ny: nyParts(ms)}; }).filter((b) => Number.isFinite(b.ms));
  if (!all.length) { box.replaceChildren(); text(statId, data ? "No candles in the last read" : "Waiting for the first chart read"); return; }
  const day = all[all.length - 1].ny.date;
  const first = all.findIndex((b) => b.ny.date === day && (chartMarket === "QQQ" || b.ny.hm >= "04:00"));
  const bars = all.slice(first);
  const ema12 = ema(all.map((b) => b.c), 2 / 13);
  let vwap = null, ema10 = null;
  if (chartMarket === "QQQ") {
    const rth = []; all.forEach((b, i) => { if (b.ny.hm >= "09:30" && b.ny.hm < "16:00") rth.push(i); });
    const e = ema(rth.map((i) => all[i].c), 2 / 11); ema10 = new Map(rth.map((i, k) => [i, e[k]]));
    let pv = 0, vol = 0; vwap = new Map();
    bars.forEach((b, k) => { if (b.ny.hm >= "09:30") { pv += ((b.h + b.l + b.c) / 3) * b.v; vol += b.v; if (vol > 0) vwap.set(first + k, pv / vol); } });
  }
  const levels = chartLevels(chartMarket), markers = chartMarkers(chartMarket, bars);
  const W = Math.max(320, box.clientWidth || 800), H = W < 600 ? 230 : 320, padL = 8, padR = 60, padT = 12, padB = 22;
  const slot = (W - padL - padR) / Math.max(bars.length, 72);
  const ymin = Math.min(...bars.map((b) => b.l), ...levels.map((x) => x.price)), ymax = Math.max(...bars.map((b) => b.h), ...levels.map((x) => x.price));
  const span = ymax - ymin || 1;
  const y = (p) => padT + ((ymax - p) / span) * (H - padT - padB), x = (k) => padL + (k + 0.5) * slot;
  const digits = chartMarket === "QQQ" ? 2 : 0;
  const svg = svgEl("svg", {viewBox: `0 0 ${W} ${H}`, class: "chart-svg", role: "img", "aria-label": `${CHART_MARKETS[chartMarket]} 5-minute candles`});
  for (let i = 0; i <= 4; i++) {
    const p = ymin + (span * i) / 4;
    svg.append(svgEl("line", {x1: padL, x2: W - padR, y1: y(p).toFixed(1), y2: y(p).toFixed(1), class: "c-grid"}));
    svg.append(svgEl("text", {x: W - padR + 6, y: (y(p) + 4).toFixed(1), class: "c-axis"}, p.toFixed(digits)));
  }
  bars.forEach((b, k) => {
    if (b.ny.hm.endsWith(":00")) {
      svg.append(svgEl("line", {x1: x(k).toFixed(1), x2: x(k).toFixed(1), y1: padT, y2: H - padB, class: "c-grid"}));
      if (x(k) > padL + 18 && (slot * 12 >= 34 || b.ny.hm.slice(0, 2) % 2 === 0)) svg.append(svgEl("text", {x: x(k).toFixed(1), y: H - 6, class: "c-axis c-mid"}, b.ny.hm));
    }
    if (b.ny.hm === "09:30") svg.append(svgEl("line", {x1: (x(k) - slot / 2).toFixed(1), x2: (x(k) - slot / 2).toFixed(1), y1: padT, y2: H - padB, class: "c-open"}));
  });
  const bw = Math.max(1, Math.min(9, slot * 0.62));
  bars.forEach((b, k) => {
    const cls = b.c >= b.o ? "c-up" : "c-dn";
    svg.append(svgEl("line", {x1: x(k).toFixed(1), x2: x(k).toFixed(1), y1: y(b.h).toFixed(1), y2: y(b.l).toFixed(1), class: cls}));
    svg.append(svgEl("rect", {x: (x(k) - bw / 2).toFixed(1), y: y(Math.max(b.o, b.c)).toFixed(1), width: bw.toFixed(1), height: Math.max(1, Math.abs(y(b.o) - y(b.c))).toFixed(1), class: cls}));
  });
  const path = (pts, cls) => { pts = pts.filter(Boolean); if (pts.length > 1) svg.append(svgEl("path", {d: pts.map(([px, py], i) => `${i ? "L" : "M"}${px.toFixed(1)} ${py.toFixed(1)}`).join(" "), class: cls})); };
  path(bars.map((b, k) => [x(k), y(ema12[first + k])]), "c-ema12");
  if (vwap) path(bars.map((b, k) => (vwap.has(first + k) ? [x(k), y(vwap.get(first + k))] : null)), "c-vwap");
  if (ema10) path(bars.map((b, k) => (ema10.has(first + k) ? [x(k), y(ema10.get(first + k))] : null)), "c-ema10");
  for (const L of levels) {
    svg.append(svgEl("line", {x1: padL, x2: W - padR, y1: y(L.price).toFixed(1), y2: y(L.price).toFixed(1), class: L.cls}));
    svg.append(svgEl("text", {x: padL + 4, y: (y(L.price) - 4).toFixed(1), class: `c-label ${L.cls}`}, L.label));
  }
  for (const m of markers) {
    const b = bars[m.index], px = x(m.index);
    svg.append(svgEl("path", {d: m.up ? `M${px.toFixed(1)} ${(y(b.l) + 6).toFixed(1)} l-5 9 h10 z` : `M${px.toFixed(1)} ${(y(b.h) - 6).toFixed(1)} l-5 -9 h10 z`, class: m.cls}));
  }
  box.replaceChildren(svg);
  const last = bars[bars.length - 1], ref = bars.find((b) => b.ny.hm >= "09:30") || bars[0];
  const bp = (last.c / ref.o - 1) * 1e4;
  text(statId, `${CHART_MARKETS[chartMarket]} ${last.c.toFixed(digits)} · ${bp >= 0 ? "+" : ""}${bp.toFixed(0)} bp since the ${ref.ny.hm} open · last candle ${last.ny.hm} NY${data.delay ? ` · ${data.delay} min late` : ""} · ${bars.length} candles`);
}

async function refreshCharts() {
  try { charts = await request("/api/charts"); drawCharts(); } catch (e) { if (e.name !== "AbortError") for (const p of CHART_PANELS) text(p.stat, `${e.message} · chart may be stale`); }
}

function renderToday(d) {
  todayData = d;
  renderStatus(d.system);
  text("today-day", `${new Date(d.day + "T12:00:00Z").toLocaleDateString("en-GB", {weekday: "long", day: "numeric", month: "long", year: "numeric"})}`.toUpperCase());
  const signal = (method, market) => d.signals.find((s) => s.method === method && s.market === market);
  text("sig-NAS100", signalText(signal("FIRST_CANDLE", "NAS100")));
  text("sig-US500", signalText(signal("FIRST_CANDLE", "US500")));
  text("sig-CONNORS", signalText(signal("CONNORS", "QQQ")));
  text("sig-DIP", signalText(signal("DIP", "QQQ")));
  for (const market of ["NAS100", "US500"]) {
    const p = (d.playbook || []).find((x) => x.market === market);
    text(`play-${market}`, playbookText(p)); $(`play-${market}`).dataset.tone = playbookTone(p);
  }
  renderTiles(d);
  const status = new Map(d.events.map((e) => [e.name, e]));
  rows("plan", d.plan, [
    (p) => EVENTS[p.event] || p.event, (p) => p.new_york, (p) => p.uk,
    (p) => { const e = status.get(p.event); return e ? `${display(e.status)}${e.note ? ` · ${display(e.note)}` : ""}` : p.qqq_delayed ? "Waiting · read after the QQQ delay" : "Waiting"; },
  ], {3: (p) => ({FAILED: "neg", MISSED: "neg", DONE: "pos"})[status.get(p.event)?.status] || ""});
  rows("cfd", d.cfd, [
    (t) => MARKETS[t.market] || t.market, (t) => side(t.side), (t) => t.amount,
    (t) => `${price(t.entry_price)} · ${ny(t.entry_at)} NY`, (t) => `${price(t.stop)}${t.worst != null ? ` · worst ${price(t.worst)}` : ""}`,
    (t) => t.exit_reason === "MISSED_EXIT" ? "Missed exit · no price" : t.exit_at ? `${price(t.exit_price)} · ${display(t.exit_reason)} ${ny(t.exit_at)} NY` : "Open",
    (t) => t.exit_at ? money(t.pnl_usd) : t.open_pnl_usd != null ? `${money(t.open_pnl_usd)} open` : "Open",
  ], {6: (t) => tone(t.exit_at ? t.pnl_usd : t.open_pnl_usd)});
  rows("crosses", d.crosses || [], [
    (t) => `${ny(new Date(Date.parse(t.signal_candle) + 300000).toISOString())} NY · QQQ ${price(t.qqq_close)} vs VWAP ${price(t.qqq_vwap)}`,
    (t) => t.amount ?? "—",
    (t) => t.entry_price == null ? `Missed entry · ${display(t.note)}` : `${price(t.entry_price)} · ${ny(t.entry_at)} NY`,
    (t) => t.entry_price == null ? "—" : t.exit_reason === "MISSED_EXIT" ? "Missed exit · no price" : t.exit_at ? `${price(t.exit_price)} · ${display(t.exit_reason)} ${ny(t.exit_at)} NY` : "Open · out back below VWAP or 16:00",
    (t) => t.exit_at ? money(t.pnl_usd) : t.open_pnl_usd != null ? `${money(t.open_pnl_usd)} open` : "Open",
  ], {4: (t) => tone(t.exit_at ? t.pnl_usd : t.open_pnl_usd)});
  rows("dips", d.dips || [], [
    (t) => `${RULES[t.rule] || t.rule}${t.book ? " · in the book" : ""}`,
    (t) => `${t.day} ${ny(t.entry_at)} NY`, (t) => t.amount,
    (t) => price(t.entry_price),
    (t) => t.exit_reason === "MISSED_EXIT" ? "Missed exit · no price" : t.exit_at ? `${price(t.exit_price)} · ${display(t.exit_reason)} ${t.exit_day} ${ny(t.exit_at)} NY` : "Open · sells on its rule at 15:50",
    (t) => t.exit_at ? money(t.pnl_usd) : "Open",
  ], {5: (t) => tone(t.pnl_usd)});
  rows("stock-orders", d.stock_orders || [], [
    (o) => o.code, (o) => price(o.close), (o) => price(o.limit_price), (o) => Number(o.crsi).toFixed(1), (o) => Number(o.adx).toFixed(1),
    (o) => o.tier === "A" ? "A · top 500" : "B · 501-1500",
    (o) => o.status === "FILLED" ? `Filled ${o.fill_day} at ${price(o.fill_price)}` : o.status === "CANCELLED" ? `Cancelled · ${display(o.note)}` : o.status === "NOT_FILLED" ? `Not filled ${o.fill_day}` : `${STOCK_STATUS[o.status] || o.status} · signal ${o.signal_day}`,
  ], {6: (o) => o.status === "FILLED" ? "pos" : ""});
  rows("stock-trades", d.stock_trades || [], [
    (t) => `${t.code} · tier ${t.tier}`, (t) => t.entry_day, (t) => price(t.entry_price),
    (t) => t.exit_day ? `${price(t.exit_price)} · ${display(t.exit_reason)} ${t.exit_day}` : "Open · sells at the first close with ConnorsRSI above 70",
    (t) => t.exit_day ? money(t.pnl_usd) : "Open",
  ], {4: (t) => tone(t.pnl_usd)});
  rows("options", d.options, [
    (o) => METHODS[o.method] || o.method,
    (o) => `${o.put_call} ${o.strike} · expires ${o.expiry}`,
    (o) => o.entry_delta == null ? "—" : Number(o.entry_delta).toFixed(2),
    (o) => `${price(o.entry_ask)} · ${ny(o.entry_for)} NY`,
    (o) => o.exit_reason === "MISSED_EXIT" ? "Missed exit · no price" : o.exit_for ? `${price(o.exit_bid)} · ${ny(o.exit_for)} NY` : `Open · sells ${o.method === "CONNORS_QQQ" ? o.exit_day + " 15:55" : "13:00"} NY`,
    (o) => o.pnl_usd == null ? (o.exit_reason ? display(o.exit_reason) : "Open") : money(o.pnl_usd),
  ], {5: (o) => tone(o.pnl_usd)});
  if (charts) drawCharts();
}

function renderRecord(d) {
  renderStatus(d.system);
  const real = d.summary.filter((m) => m.method !== "DIP BOOK");
  const total = real.reduce((s, m) => s + (m.total_usd ?? 0), 0), closed = real.reduce((s, m) => s + (m.closed ?? 0), 0), wins = real.reduce((s, m) => s + (m.wins ?? 0), 0);
  text("rec-total", money(total)); $("rec-total").className = `tile-value ${tone(total)}`;
  text("rec-closed", closed); text("rec-closed-sub", `${real.reduce((s, m) => s + (m.trades ?? 0), 0)} recorded, ${real.reduce((s, m) => s + ((m.trades ?? 0) - (m.closed ?? 0)), 0)} open or without a price`);
  text("rec-win", closed ? `${Math.round((100 * wins) / closed)}%` : "—"); text("rec-win-sub", closed ? `${wins} of ${closed} closed trades` : "no closed trades yet");
  const since = real.map((m) => m.first_day).filter(Boolean).sort()[0];
  text("rec-since", since || "—");
  rows("summary", d.summary, [
    (m) => methodName(m.method), (m) => m.trades, (m) => m.closed, (m) => m.wins ?? 0,
    (m) => money(m.total_usd), (m) => money(m.mean_usd), (m) => m.first_day,
  ], {4: (m) => tone(m.total_usd)});
  const cell = (k) => (b) => { const c = b.cells[k]; return c ? `${c.trades} · ${money(c.total_usd)}${c.closed ? ` (avg ${money(c.mean_usd)})` : " · none closed"}` : "—"; };
  const books = (d.books || []).filter((b) => Object.keys(b.cells).length > 0);
  rows("books", books, [(b) => BOOKS[b.book] || b.book, cell("NAS100"), cell("US500"), cell("QQQ option"), cell("CROSS")],
    {1: (b) => tone(b.cells.NAS100?.total_usd), 2: (b) => tone(b.cells.US500?.total_usd), 3: (b) => tone(b.cells["QQQ option"]?.total_usd), 4: (b) => tone(b.cells.CROSS?.total_usd)});
  const when = (iso) => !iso ? "—" : /^\d{4}-\d{2}-\d{2}$/.test(iso) ? `${iso} close` : `${ny(iso)} NY`;
  rows("trades", d.trades, [
    (t) => t.day, (t) => methodName(t.method), (t) => side(t.side),
    (t) => when(t.opened), (t) => t.exit_reason === "MISSED_EXIT" ? "Missed exit" : t.closed ? when(t.closed) : "Open",
    (t) => display(t.exit_reason) || "—", (t) => money(t.pnl_usd),
  ], {6: (t) => tone(t.pnl_usd)});
  $("prev").disabled = offset === 0;
  $("next").disabled = !d.has_more;
}

function renderSystem(s) {
  renderStatus(s);
  text("auth-state", `Saxo login: ${s.oauth}${s.oauth_failure ? ` · ${display(s.oauth_failure)}` : ""}`);
  const facts = [
    ["Environment", s.data_environment], ["Build", s.build], ["Worker", display(s.worker_health)],
    ["Saxo session level", s.trade_level ? `${s.trade_level} (the app never changes it)` : "Not read yet"],
    ["Login renewal", s.login_minutes_left == null ? "No login" : `${s.login_minutes_left} min before the next renewal is due`],
    ["QQQ delay", s.qqq_delay_minutes == null ? "Not measured yet" : `${s.qqq_delay_minutes} min`],
    ["Calendar", s.calendar_loaded ? `Loaded for ${s.calendar_loaded}` : "Not loaded"],
    ["Saxo requests since start", s.saxo_calls],
  ];
  $("facts").replaceChildren(...facts.map(([k, v]) => {
    const div = document.createElement("div"), dt = document.createElement("dt"), dd = document.createElement("dd");
    dt.textContent = k; dd.textContent = String(v ?? "—"); div.append(dt, dd); return div;
  }));
  if ($("system-detail").open) text("system-json", JSON.stringify(s, null, 2));
}

async function refresh(force = false) {
  if (document.hidden || (pending && !force)) return;
  if (force) pending?.abort();
  const controller = new AbortController(); pending = controller;
  try {
    const path = page === "record" ? `/api/record?offset=${offset}` : page === "system" ? "/api/system" : "/api/today";
    const d = await request(path, {signal: controller.signal});
    if (controller.signal.aborted) return;
    if (page === "record") renderRecord(d);
    else if (page === "system") renderSystem(d);
    else renderToday(d);
    lastRefresh = new Date().toISOString();
    text("notice", "");
    text("last-refresh", `Refreshed ${uk(lastRefresh)} UK`);
  } catch (e) {
    if (e.name !== "AbortError") text("notice", `${e.message} · displayed values may be stale`);
  } finally {
    if (pending === controller) pending = null;
  }
}

show(page, true);
for (const a of document.querySelectorAll("nav a")) if ((a.getAttribute("href").replace(/^\//, "") || "today") === page) a.setAttribute("aria-current", "page");
$("saxo-connect").onsubmit = async (event) => {
  event.preventDefault();
  const button = event.currentTarget.querySelector("button"); button.disabled = true;
  text("oauth-notice", "");
  try { const result = await request("/oauth/saxo/start", {method: "POST", timeout: 15000}); window.location.assign(result.authorization_url); }
  catch (e) { text("oauth-notice", e.message); button.disabled = false; }
};
window.addEventListener("resize", () => { if (page === "today" && charts) drawCharts(); });
if (page === "today") { refreshCharts(); setInterval(refreshCharts, 60000); }
$("prev").onclick = () => { offset = Math.max(0, offset - 100); refresh(true); };
$("next").onclick = () => { offset += 100; refresh(true); };
$("system-detail").addEventListener("toggle", () => refresh(true));
document.addEventListener("visibilitychange", () => { if (document.hidden) pending?.abort(); else refresh(true); });
setInterval(() => {
  const now = new Date().toISOString();
  text("clock", `${ny(now)} NY · ${uk(now)} UK`);
  if (lastRefresh && Date.now() - Date.parse(lastRefresh) > (REFRESH_SECONDS * 3 + 5) * 1000) text("last-refresh", `Stale · last refresh ${uk(lastRefresh)} UK`);
}, 1000);
setInterval(() => refresh(), REFRESH_SECONDS * 1000);
refresh(true);
