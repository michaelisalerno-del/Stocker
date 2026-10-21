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
  STOCK_DIPS: "Stock dips · after-close scan",
};
const MARKETS = {NAS100: "Nasdaq CFD", US500: "S&P CFD"};
const METHODS = {"CFD NAS100": "Nasdaq CFD · first candle", "CFD US500": "S&P CFD · first candle", FIRST_CANDLE_QQQ: "QQQ option · first candle", CONNORS_QQQ: "QQQ call · Connors", "CROSS NAS100": "Nasdaq CFD · VWAP crossings", "DIP CONNORS": "Nasdaq CFD · Connors (days)", "DIP TURNAROUND": "Nasdaq CFD · Turnaround Tuesday", "DIP FEAR": "Nasdaq CFD · fear rule (3 down closes, 30-day IV ≥ 30%)", "DIP BOOK": "Nasdaq CFD · dip book, one at a time", "STOCK DIP": "US stocks · Connors pullback (8% limit, out above ConnorsRSI 70)"};
const STOCK_STATUS = {PENDING: "Resting for the next session", NOT_FILLED: "Not filled", FILLED: "Filled", CANCELLED: "Cancelled"};
const methodName = (m) => METHODS[m] || (m.startsWith("STOCK DIP ") ? `${METHODS["STOCK DIP"]} · ${m.slice(10)}` : m);
const RULES = {CONNORS: "Connors", TURNAROUND: "Turnaround Tuesday", FEAR: "Fear rule"};
const BOOKS = {PULLBACK: "Pullback cell · long signal, prior close below the 50-day", WEEKDAYS: "Tuesday to Friday", AGREE: "Nasdaq and S&P signals agree", "NO OVERSOLD SHORTS": "Shorts skipped after an oversold close (3+ down closes or RSI2 < 10)", "AFTER A +1% DAY": "After a +1% day", PLAYBOOK: "Playbook · the call in the pullback cell, the Nasdaq CFD otherwise, no Monday or oversold shorts, S&P in the cell only", "SIZED V2": "Sized · playbook units (cell 2, other longs 1, shorts 1 unless Monday or oversold)", "SIZED V3": "Sized · score units 0-3 (+cell, +after a +1% day, +agree, −Monday, −oversold short)", "STOP 75 BP": "Nasdaq CFD · what a 75 bp stop at the same $200 risk would have done", "CROSS PULLBACK": "VWAP crossings · pullback days", "CROSS BEFORE 14:00": "VWAP crossings · entered before 14:00"};
// The protocol's fixed checkpoints: each is judged once, when its count is reached (PAPER-RECORD-PROTOCOL.md
// and amendments 2-5). Counts are read from the Record summary; nothing here judges anything.
const VERDICTS = [
  {label: "A · CFD costs · Nasdaq", target: 40, note: "closed Nasdaq CFD trades · as assumed if the mean round-trip spread is at most 0.93 bp", count: (c) => c.closed("CFD NAS100")},
  {label: "A · CFD costs · S&P", target: 40, note: "closed S&P CFD trades · as assumed if the mean round-trip spread is at most 1.80 bp", count: (c) => c.closed("CFD US500")},
  {label: "B · QQQ option spread", target: 40, note: "closed first-candle options · as assumed if the mean half-spread is at most 1.8%", count: (c) => c.closed("FIRST_CANDLE_QQQ")},
  {label: "C · QQQ option carry", target: 60, note: "closed first-candle options · viable if the mean carry is above −5 bp per unit of delta", count: (c) => c.closed("FIRST_CANDLE_QQQ")},
  {label: "D · Direction breakage check", target: 250, note: "Nasdaq CFD days · broken only if the mean day is below zero with t < −2", count: (c) => c.trades("CFD NAS100")},
  {label: "E · Connors calls · first look", target: 40, note: "Connors call trades · reported, not judged", count: (c) => c.trades("CONNORS_QQQ")},
  {label: "VWAP crossings · first look", target: 100, note: "closed crossings · mean result and the spread paid, against 1 bp", count: (c) => c.closed("CROSS NAS100")},
  {label: "Dip rules · first look", target: 40, note: "closed dip trades · mean result after financing, spread against 0.7 bp", count: (c) => c.closed("DIP CONNORS") + c.closed("DIP TURNAROUND") + c.closed("DIP FEAR")},
  {label: "Fear rule · first look", target: 10, note: "fear-rule trades", count: (c) => c.trades("DIP FEAR")},
  {label: "Pullback book · first look", target: 40, note: "closed Nasdaq CFD trades in the pullback cell", count: (c) => c.book("PULLBACK", "NAS100")},
  {label: "Stock dips · first look", target: 100, note: "filled stock trades · against +3.4% a trade and 68% winners", count: (c) => c.trades("STOCK DIP")},
];
const USD = new Intl.NumberFormat("en-US", {style: "currency", currency: "USD"});
let offset = 0, pending = null, lastRefresh = "", serverOffset = 0;

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
const dayName = (iso) => iso ? new Date(iso + "T12:00:00Z").toLocaleDateString("en-GB", {weekday: "short", day: "numeric", month: "short"}) : "—";
const join = (parts) => parts.filter(Boolean).join(" · ");
const el = (tag, className, content) => { const e = document.createElement(tag); if (className) e.className = className; if (content != null) e.textContent = content; return e; };
const serverNow = () => Date.now() + serverOffset;
const countdown = (iso) => {
  const ms = Date.parse(iso) - serverNow();
  if (!(ms > 0)) return "due now";
  const h = Math.floor(ms / 3600000), m = Math.floor((ms % 3600000) / 60000);
  return `in ${h ? `${h}h ` : ""}${m}m`;
};

/** Fill a table body. A column gives a string, a Node, or {main, sub} for a two-line cell; `classes`
 * adds a class per column index, `rowClass` one per row. The empty-state paragraph `<id>-empty` and the
 * table's scroll box swap on emptiness. */
function rows(id, items, columns, classes = {}, rowClass = null) {
  const body = $(id);
  const heads = [...body.closest("table").querySelectorAll("thead th")];
  body.replaceChildren(...items.map((item) => {
    const tr = document.createElement("tr");
    const cls = rowClass?.(item);
    if (cls) tr.className = cls;
    columns.forEach((f, i) => {
      const td = document.createElement("td");
      td.dataset.label = heads[i]?.textContent || "";
      if (heads[i]?.classList.contains("num")) td.classList.add("num");
      if (heads[i]?.classList.contains("bar")) td.classList.add("bar");
      const value = f(item);
      if (value instanceof Node) td.append(value);
      else if (value && typeof value === "object") { td.append(document.createTextNode(String(value.main ?? "—"))); if (value.sub) td.append(el("span", "sub", value.sub)); }
      else td.textContent = String(value ?? "—");
      const c = classes[i]?.(item);
      if (c) td.classList.add(c);
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

// status --------------------------------------------------------------------------------------
function renderStatus(s) {
  if (s.server_time) serverOffset = Date.parse(s.server_time) - Date.now();
  const login = s.oauth === "AUTHENTICATED";
  text("login-chip", login ? `Saxo connected · renews in ${s.login_minutes_left ?? "?"} min` : `Saxo: ${display(s.oauth)}`);
  $("login-chip").className = login ? "chip" : "chip bad";
  text("day-chip", s.session ? `US session ${ny(s.session.open)}–${ny(s.session.close)} NY${s.session.regular ? "" : " · early close, skipped"}` : "No US session today");
  const late = s.qqq_delay_minutes;
  text("delay-chip", late == null ? "QQQ delay not measured yet" : late ? `QQQ ${late} min late` : "QQQ live");
  $("delay-chip").className = late ? "chip warn" : "chip";
  const problems = Object.entries(s.problems || {});
  show("problem-chip", problems.length > 0);
  text("problem-chip", problems.map(([k, v]) => `${display(k)}: ${display(v)}`).join(" · "));
  $("problem-chip").className = "chip bad";
  $("status").dataset.tone = !login || s.worker_health !== "RUNNING" ? "bad" : problems.length ? "warn" : "ok";
}

// today ---------------------------------------------------------------------------------------
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
const openMark = (t) => t.exit_at ? null : t.open_pnl_usd;

function renderTiles(d) {
  const s = d.system, status = new Map(d.events.map((e) => [e.name, e]));
  const closed = [...d.cfd, ...(d.crosses || []), ...(d.options || []).filter((o) => o.exit_day === d.day), ...(d.dips || []).filter((t) => t.exit_day === d.day), ...(d.stock_trades || []).filter((t) => t.exit_day === d.day)]
    .reduce((sum, t) => sum + (t.pnl_usd ?? 0), 0);
  const open = [...d.cfd, ...(d.crosses || []), ...(d.dips || [])].reduce((sum, t) => sum + (openMark(t) ?? 0), 0);
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
  text("tile-feed-sub", `CFD quotes live · Saxo renews in ${s.login_minutes_left ?? "?"} min · ${problems ? `${problems} problem${problems > 1 ? "s" : ""}` : "no problems"}`);
  return next;
}

// charts (Today) --------------------------------------------------------------------------------
const CHART_MARKETS = {NAS100: "Nasdaq CFD", US500: "S&P CFD", QQQ: "QQQ"};
const CHART_PANELS = [{box: "chart", stat: "chart-stat", head: "chart-price", readout: "chart-readout", market: "QQQ"}, {box: "chart-US500", stat: "chart-stat-US500", head: "chart-price-US500", readout: "chart-readout-US500", market: "US500"}];
let charts = null, todayData = null, lastLevelsKey = "";
const SVG = "http://www.w3.org/2000/svg";
const svgEl = (tag, attrs = {}, content) => { const e = document.createElementNS(SVG, tag); for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v); if (content != null) e.textContent = content; return e; };
const nyFormat = new Intl.DateTimeFormat("en-CA", {timeZone: "America/New_York", year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hourCycle: "h23"});
function nyParts(ms) {
  const p = {}; for (const {type, value} of nyFormat.formatToParts(new Date(ms))) p[type] = value;
  return {date: `${p.year}-${p.month}-${p.day}`, hm: `${p.hour}:${p.minute}`};
}
const ema = (values, alpha) => { let v = null; return values.map((x) => (v = v == null ? x : v + alpha * (x - v))); };
const slot5 = (iso) => Math.floor(Date.parse(iso) / 300000) * 300000;
/** Clean axis ticks: a 1 / 2 / 2.5 / 5 step covering the range in about `count` steps. */
function niceTicks(min, max, count = 4) {
  const raw = (max - min || 1) / count, mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) || raw;
  const digits = Math.max(0, -Math.floor(Math.log10(step)));
  const ticks = []; for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-6; v += step) ticks.push(Number(v.toFixed(digits)));
  return {ticks, digits};
}
/** Sum two label positions apart: a level whose label would sit on the previous one's goes under its line. */
function placeLabels(levels, y) {
  let previous = -1e9;
  return [...levels].sort((a, b) => y(a.price) - y(b.price)).map((L) => {
    let at = y(L.price) - 4;
    if (at - previous < 12) at = y(L.price) + 11;
    previous = at;
    return {...L, labelY: at};
  });
}
/** A crosshair over `bars`: the nearest candle's readout goes to `readoutId`. `describe(bar, index)` makes the text. */
function crosshair(svg, bars, geometry, readoutId, describe) {
  const {padL, padT, plotW, plotH, slot, x} = geometry;
  const line = svgEl("line", {class: "c-cross", y1: padT, y2: padT + plotH, x1: 0, x2: 0, visibility: "hidden"});
  const hit = svgEl("rect", {class: "c-hit", x: padL, y: padT, width: plotW, height: plotH});
  svg.append(line, hit);
  const point = svg.createSVGPoint();
  hit.addEventListener("pointermove", (event) => {
    const ctm = svg.getScreenCTM(); if (!ctm) return;
    point.x = event.clientX; point.y = event.clientY;
    const local = point.matrixTransform(ctm.inverse());
    const k = Math.max(0, Math.min(bars.length - 1, Math.floor((local.x - padL) / slot)));
    line.setAttribute("x1", x(k).toFixed(1)); line.setAttribute("x2", x(k).toFixed(1)); line.setAttribute("visibility", "visible");
    text(readoutId, describe(bars[k], k));
  });
  hit.addEventListener("pointerleave", () => { line.setAttribute("visibility", "hidden"); text(readoutId, ""); });
}

function chartLevels(chartMarket) {
  const d = todayData; if (!d) return [];
  if (chartMarket === "QQQ") {
    return (d.options || []).filter((o) => o.method === "FIRST_CANDLE_QQQ").map((o) => ({price: o.strike, cls: "c-strike", label: `${o.put_call.toLowerCase()} ${o.strike}`}));
  }
  const t = d.cfd.find((x) => x.market === chartMarket); if (!t) return [];
  const out = [{price: t.entry_price, cls: "c-entry", label: `entry ${side(t.side).toLowerCase()} ${price(t.entry_price)}`}];
  if (t.exit_at && t.exit_price != null) out.push({price: t.exit_price, cls: "c-exit", label: `exit ${display(t.exit_reason)} ${price(t.exit_price)}`});
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
  for (const p of CHART_PANELS) drawChart(p);
}

function drawChart({market: chartMarket, box: boxId, stat: statId, head: headId, readout: readoutId}) {
  const box = $(boxId); if (!box) return;
  const data = charts?.markets?.[chartMarket];
  const all = (data?.bars || []).map((b) => { const ms = Date.parse(b[0]); return {ms, o: b[1], h: b[2], l: b[3], c: b[4], v: b[5], ny: nyParts(ms)}; }).filter((b) => Number.isFinite(b.ms));
  if (!all.length) { box.replaceChildren(); text(statId, data ? "No candles in the last read" : "Waiting for the first chart read"); text(headId, ""); return; }
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
  const W = Math.max(320, box.clientWidth || 800), H = W < 600 ? 230 : 320, padL = 8, padR = 58, padT = 12, padB = 22;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const slot = plotW / Math.max(bars.length, 72);
  const lo = Math.min(...bars.map((b) => b.l), ...levels.map((x) => x.price)), hi = Math.max(...bars.map((b) => b.h), ...levels.map((x) => x.price));
  const pad = (hi - lo || 1) * 0.04, ymin = lo - pad, ymax = hi + pad, span = ymax - ymin;
  const y = (p) => padT + ((ymax - p) / span) * plotH, x = (k) => padL + (k + 0.5) * slot;
  const digits = chartMarket === "QQQ" ? 2 : 0;
  const svg = svgEl("svg", {viewBox: `0 0 ${W} ${H}`, class: "chart-svg", role: "img", "aria-label": `${CHART_MARKETS[chartMarket]} 5-minute candles`});
  const axis = niceTicks(ymin, ymax, 5);
  for (const p of axis.ticks) {
    svg.append(svgEl("line", {x1: padL, x2: W - padR, y1: y(p).toFixed(1), y2: y(p).toFixed(1), class: "c-grid"}));
    svg.append(svgEl("text", {x: W - padR + 6, y: (y(p) + 4).toFixed(1), class: "c-axis"}, p.toFixed(Math.max(digits === 0 ? 0 : axis.digits, 0))));
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
  for (const L of placeLabels(levels, y)) {
    svg.append(svgEl("line", {x1: padL, x2: W - padR, y1: y(L.price).toFixed(1), y2: y(L.price).toFixed(1), class: L.cls}));
    // A surface-coloured backing keeps the label legible where it lands on candles.
    svg.append(svgEl("rect", {x: padL + 1, y: (L.labelY - 9.5).toFixed(1), width: (L.label.length * 6.4 + 6).toFixed(1), height: 12, rx: 2, class: "c-label-bg"}));
    svg.append(svgEl("text", {x: padL + 4, y: L.labelY.toFixed(1), class: `c-label ${L.cls}`}, L.label));
  }
  for (const m of markers) {
    const b = bars[m.index], px = x(m.index);
    svg.append(svgEl("path", {d: m.up ? `M${px.toFixed(1)} ${(y(b.l) + 6).toFixed(1)} l-5 9 h10 z` : `M${px.toFixed(1)} ${(y(b.h) - 6).toFixed(1)} l-5 -9 h10 z`, class: m.cls}));
  }
  const f = (v) => v == null ? "—" : v.toFixed(digits);
  crosshair(svg, bars, {padL, padT, plotW, plotH, slot, x}, readoutId, (b, k) => join([
    `${b.ny.hm} NY`, `O ${f(b.o)} H ${f(b.h)} L ${f(b.l)} C ${f(b.c)}`, `EMA12 ${f(ema12[first + k])}`,
    vwap && vwap.has(first + k) ? `VWAP ${f(vwap.get(first + k))}` : "", ema10 && ema10.has(first + k) ? `EMA10 ${f(ema10.get(first + k))}` : "",
    chartMarket === "QQQ" ? `vol ${Math.round(b.v).toLocaleString("en-US")}` : "",
  ]));
  box.replaceChildren(svg);
  const last = bars[bars.length - 1], ref = bars.find((b) => b.ny.hm >= "09:30") || bars[0];
  const bp = (last.c / ref.o - 1) * 1e4;
  const head = $(headId);
  if (head) {
    const delta = el("span", `delta ${tone(bp)}`, `${bp >= 0 ? "+" : ""}${bp.toFixed(0)} bp since ${ref.ny.hm}`);
    head.replaceChildren(document.createTextNode(`${last.c.toFixed(digits)} `), delta);
  }
  text(statId, join([`last candle ${last.ny.hm} NY`, data.delay ? `${data.delay} min late` : "", `${bars.length} candles`]));
}

async function refreshCharts() {
  try { charts = await request("/api/charts"); drawCharts(); } catch (e) { if (e.name !== "AbortError") for (const p of CHART_PANELS) text(p.stat, `${e.message} · chart may be stale`); }
}

function methodSums(d) {
  const sum = (items, pick) => items.reduce((s, t) => s + (pick(t) ?? 0), 0);
  const closedOf = (items) => sum(items.filter((t) => t.pnl_usd != null), (t) => t.pnl_usd);
  const count = (n, word) => n ? `${n} ${word}${n === 1 ? "" : "s"}` : "";
  const cfd = d.cfd, open = sum(cfd, openMark);
  text("cfd-sum", join([count(cfd.length, "trade"), cfd.some((t) => t.pnl_usd != null) ? `closed ${money(closedOf(cfd))}` : "", cfd.some((t) => openMark(t) != null) ? `open ${money(open)}` : ""]));
  const options = d.options || [];
  text("options-sum", join([count(options.length, "trade"), count(options.filter((o) => !o.exit_read).length, "open"), options.some((o) => o.pnl_usd != null) ? money(closedOf(options)) : ""]));
  const crosses = d.crosses || [], traded = crosses.filter((t) => t.entry_price != null);
  text("crosses-sum", join([count(traded.length, "trade"), count(crosses.length - traded.length, "missed entry"), traded.some((t) => t.pnl_usd != null) ? money(closedOf(traded)) : "", traded.some((t) => openMark(t) != null) ? `open ${money(sum(traded, openMark))}` : ""]));
  const dips = d.dips || [];
  text("dips-sum", join([count(dips.filter((t) => !t.exit_at).length, "open"), count(dips.filter((t) => t.exit_at).length, "closed"), dips.some((t) => t.pnl_usd != null) ? money(closedOf(dips)) : "", dips.some((t) => openMark(t) != null) ? `open ${money(sum(dips, openMark))}` : ""]));
  const orders = d.stock_orders || [], stocks = d.stock_trades || [];
  text("stocks-sum", join([count(orders.filter((o) => o.status === "PENDING").length, "resting order"), count(orders.filter((o) => o.status === "FILLED" && o.fill_day === d.day).length, "fill today"), count(stocks.filter((t) => !t.exit_day).length, "open position"), stocks.some((t) => t.pnl_usd != null && t.exit_day === d.day) ? `closed today ${money(closedOf(stocks.filter((t) => t.exit_day === d.day)))}` : ""]));
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
  const next = renderTiles(d);
  const status = new Map(d.events.map((e) => [e.name, e]));
  const done = d.plan.filter((p) => status.get(p.event)?.status === "DONE").length;
  text("plan-progress", d.plan.length ? `${done} of ${d.plan.length} done` : "");
  rows("plan", d.plan, [
    (p) => EVENTS[p.event] || p.event, (p) => p.new_york, (p) => p.uk,
    (p) => { const e = status.get(p.event); if (e && e.status !== "WAITING") return `${display(e.status)}${e.note ? ` · ${display(e.note)}` : ""}`; if (p === next) return `Next · ${countdown(p.due)}`; return e ? `Waiting · reads at ${ny(e.effective)} NY` : p.qqq_delayed ? "Waiting · read after the QQQ delay" : "Waiting"; },
  ], {3: (p) => ({FAILED: "neg", MISSED: "neg", DONE: "pos"})[status.get(p.event)?.status] || (p === next ? "live" : "")}, (p) => p === next ? "next" : status.get(p.event)?.status === "DONE" ? "done" : "");
  rows("cfd", d.cfd, [
    (t) => MARKETS[t.market] || t.market, (t) => side(t.side), (t) => t.amount,
    (t) => `${price(t.entry_price)} · ${ny(t.entry_at)} NY`, (t) => `${price(t.stop)}${t.worst != null ? ` · worst ${price(t.worst)}` : ""}`,
    (t) => t.exit_reason === "MISSED_EXIT" ? "Missed exit · no price" : t.exit_at ? `${price(t.exit_price)} · ${display(t.exit_reason)} ${ny(t.exit_at)} NY` : "Open",
    (t) => t.exit_at ? money(t.pnl_usd) : t.open_pnl_usd != null ? `${money(t.open_pnl_usd)} open` : "Open",
  ], {6: (t) => tone(t.exit_at ? t.pnl_usd : t.open_pnl_usd)});
  rows("options", d.options, [
    (o) => METHODS[o.method] || o.method,
    (o) => `${o.put_call} ${o.strike} · expires ${o.expiry}`,
    (o) => o.entry_delta == null ? "—" : Number(o.entry_delta).toFixed(2),
    (o) => `${price(o.entry_ask)} · ${ny(o.entry_for)} NY`,
    (o) => o.exit_reason === "MISSED_EXIT" ? "Missed exit · no price" : o.exit_for ? `${price(o.exit_bid)} · ${ny(o.exit_for)} NY` : `Open · sells ${o.method === "CONNORS_QQQ" ? o.exit_day + " 15:55" : "13:00"} NY`,
    (o) => o.pnl_usd == null ? (o.exit_reason ? display(o.exit_reason) : "Open") : money(o.pnl_usd),
  ], {5: (o) => tone(o.pnl_usd)});
  rows("crosses", d.crosses || [], [
    (t) => `${ny(new Date(Date.parse(t.signal_candle) + 300000).toISOString())} NY · QQQ ${price(t.qqq_close)} vs VWAP ${price(t.qqq_vwap)}`,
    (t) => t.amount ?? "—",
    (t) => t.entry_price == null ? `Missed entry · ${display(t.note)}` : `${price(t.entry_price)} · ${ny(t.entry_at)} NY`,
    (t) => t.entry_price == null ? "—" : t.exit_reason === "MISSED_EXIT" ? "Missed exit · no price" : t.exit_at ? `${price(t.exit_price)} · ${display(t.exit_reason)} ${ny(t.exit_at)} NY` : "Open · out back below VWAP or 16:00",
    (t) => t.exit_at ? money(t.pnl_usd) : t.open_pnl_usd != null ? `${money(t.open_pnl_usd)} open` : t.entry_price == null ? "—" : "Open",
  ], {4: (t) => tone(t.exit_at ? t.pnl_usd : t.open_pnl_usd)});
  rows("dips", d.dips || [], [
    (t) => `${RULES[t.rule] || t.rule}${t.book ? " · in the book" : ""}`,
    (t) => `${t.day} ${ny(t.entry_at)} NY`, (t) => t.amount,
    (t) => price(t.entry_price),
    (t) => t.exit_reason === "MISSED_EXIT" ? "Missed exit · no price" : t.exit_at ? `${price(t.exit_price)} · ${display(t.exit_reason)} ${t.exit_day} ${ny(t.exit_at)} NY` : "Open · sells on its rule at 15:50",
    (t) => t.exit_at ? money(t.pnl_usd) : t.open_pnl_usd != null ? `${money(t.open_pnl_usd)} open` : "Open",
  ], {5: (t) => tone(t.exit_at ? t.pnl_usd : t.open_pnl_usd)});
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
  methodSums(d);
  // The chart's levels and markers come from the trades: redraw only when they change, not every refresh.
  const key = JSON.stringify([d.cfd, d.crosses, d.options]);
  if (charts && key !== lastLevelsKey) { lastLevelsKey = key; drawCharts(); }
}

// record --------------------------------------------------------------------------------------
function drawEquity(byDay) {
  const box = $("equity"); if (!box) return;
  const days = (byDay || []).filter((r) => r.pnl_usd != null);
  if (days.length < 2) { box.replaceChildren(); text("equity-stat", days.length ? "One day of closed trades: the curve starts with the second" : "No closed trades yet"); return; }
  let cum = 0, peak = 0, drawdown = 0;
  const points = days.map((r) => { cum += r.pnl_usd; peak = Math.max(peak, cum); drawdown = Math.min(drawdown, cum - peak); return {day: r.day, pnl: r.pnl_usd, closed: r.closed, cum}; });
  const W = Math.max(320, box.clientWidth || 800), H = W < 600 ? 200 : 240, padL = 8, padR = 64, padT = 16, padB = 24;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const lo = Math.min(0, ...points.map((p) => p.cum)), hi = Math.max(0, ...points.map((p) => p.cum));
  const pad = (hi - lo || 1) * 0.08, ymin = lo - pad, ymax = hi + pad, span = ymax - ymin;
  const y = (v) => padT + ((ymax - v) / span) * plotH, slot = plotW / points.length, x = (k) => padL + (k + 0.5) * slot;
  const svg = svgEl("svg", {viewBox: `0 0 ${W} ${H}`, class: "chart-svg", role: "img", "aria-label": "Cumulative closed paper P&L by day"});
  for (const v of niceTicks(ymin, ymax, 4).ticks) {
    svg.append(svgEl("line", {x1: padL, x2: W - padR, y1: y(v).toFixed(1), y2: y(v).toFixed(1), class: v === 0 ? "c-zero" : "c-grid"}));
    svg.append(svgEl("text", {x: W - padR + 6, y: (y(v) + 4).toFixed(1), class: "c-axis"}, USD.format(v).replace(/\.00$/, "")));
  }
  const every = Math.max(1, Math.ceil(points.length / (W < 600 ? 3 : 6)));
  points.forEach((p, k) => { if (k % every === 0 || k === points.length - 1) svg.append(svgEl("text", {x: x(k).toFixed(1), y: H - 6, class: `c-axis ${k === points.length - 1 ? "c-end" : k ? "c-mid" : ""}`}, dayName(p.day))); });
  const line = points.map((p, k) => `${k ? "L" : "M"}${x(k).toFixed(1)} ${y(p.cum).toFixed(1)}`).join(" ");
  const last = points[points.length - 1], neg = last.cum < 0;
  svg.append(svgEl("path", {d: `${line} L${x(points.length - 1).toFixed(1)} ${y(0).toFixed(1)} L${x(0).toFixed(1)} ${y(0).toFixed(1)} Z`, class: `c-area${neg ? " c-area-neg" : ""}`}));
  svg.append(svgEl("path", {d: line, class: `c-line${neg ? " c-line-neg" : ""}`}));
  svg.append(svgEl("circle", {cx: x(points.length - 1).toFixed(1), cy: y(last.cum).toFixed(1), r: 4, class: `c-dot${neg ? " c-dot-neg" : ""}`}));
  // The end label sits on the side the line does not come from: above after a rise, below after a fall.
  const rising = last.cum >= points[points.length - 2].cum;
  svg.append(svgEl("text", {x: (x(points.length - 1) - 8).toFixed(1), y: (rising ? y(last.cum) - 8 : y(last.cum) + 14).toFixed(1), class: "c-axis c-end"}, money(last.cum)));
  crosshair(svg, points, {padL, padT, plotW, plotH, slot, x}, "equity-readout", (p) => join([dayName(p.day), `day ${money(p.pnl)} (${p.closed} closed)`, `cumulative ${money(p.cum)}`]));
  box.replaceChildren(svg);
  const best = points.reduce((a, b) => (b.pnl > a.pnl ? b : a)), worst = points.reduce((a, b) => (b.pnl < a.pnl ? b : a));
  text("equity-stat", join([`${points.length} days with closed trades`, `best ${money(best.pnl)} (${dayName(best.day)})`, `worst ${money(worst.pnl)} (${dayName(worst.day)})`, `max drawdown ${money(drawdown)}`]));
}

function renderVerdicts(summary, books) {
  const by = new Map(summary.map((m) => [m.method, m]));
  const counts = {
    closed: (m) => by.get(m)?.closed ?? 0,
    trades: (m) => by.get(m)?.trades ?? 0,
    book: (b, instrument) => (books || []).find((x) => x.book === b)?.cells?.[instrument]?.closed ?? 0,
  };
  $("verdicts").replaceChildren(...VERDICTS.map((v) => {
    const n = v.count(counts), due = n >= v.target;
    const li = el("li", "meter");
    if (due) li.dataset.due = "1";
    const bar = document.createElement("progress"); bar.max = v.target; bar.value = Math.min(n, v.target);
    li.append(el("span", "meter-label", v.label), el("span", "meter-count", due ? `${n} · reached` : `${n} of ${v.target}`), bar, el("span", "meter-note", v.note));
    return li;
  }));
}

function renderRecord(d) {
  renderStatus(d.system);
  const real = d.summary.filter((m) => m.method !== "DIP BOOK");
  const total = real.reduce((s, m) => s + (m.total_usd ?? 0), 0), closed = real.reduce((s, m) => s + (m.closed ?? 0), 0), wins = real.reduce((s, m) => s + (m.wins ?? 0), 0);
  text("rec-total", money(total)); $("rec-total").className = `tile-value ${tone(total)}`;
  text("rec-closed", closed); text("rec-closed-sub", `${real.reduce((s, m) => s + (m.trades ?? 0), 0)} recorded, ${real.reduce((s, m) => s + ((m.trades ?? 0) - (m.closed ?? 0)), 0)} open or without a price`);
  text("rec-win", closed ? `${Math.round((100 * wins) / closed)}%` : "—"); text("rec-win-sub", closed ? `${wins} of ${closed} closed trades` : "no closed trades yet");
  const since = real.map((m) => m.first_day).filter(Boolean).sort()[0];
  text("rec-since", since ? dayName(since) : "—"); text("rec-since-sub", since ? `${since} · first recorded trade` : "first recorded trade");
  if (offset === 0) { drawEquity(d.by_day); renderVerdicts(d.summary, d.books); }
  const scale = Math.max(1, ...d.summary.map((m) => Math.abs(m.total_usd ?? 0)));
  rows("summary", d.summary, [
    (m) => methodName(m.method), (m) => m.trades, (m) => m.closed, (m) => m.wins ?? 0,
    (m) => money(m.total_usd),
    (m) => { // a signed bar from the column's midpoint; scaled to the largest total
      const v = m.total_usd ?? 0, svg = svgEl("svg", {viewBox: "0 0 120 10", preserveAspectRatio: "none", "aria-hidden": "true"});
      const w = (60 * Math.abs(v)) / scale;
      svg.append(svgEl("line", {x1: 60, x2: 60, y1: 0, y2: 10, class: "c-zero"}));
      if (w > 0) svg.append(svgEl("rect", {x: v < 0 ? (60 - w).toFixed(1) : 60, y: 1, width: Math.max(1, w).toFixed(1), height: 8, rx: 2, class: tone(v)}));
      return svg;
    },
    (m) => money(m.mean_usd), (m) => m.first_day,
  ], {4: (m) => tone(m.total_usd)});
  const cell = (k) => (b) => { const c = b.cells[k]; return c ? {main: `${c.trades} · ${money(c.total_usd)}`, sub: c.closed ? `avg ${money(c.mean_usd)} · ${c.closed} closed` : "none closed"} : "—"; };
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

// system --------------------------------------------------------------------------------------
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
  const problems = Object.entries(s.problems || {});
  $("problems").replaceChildren(...problems.map(([area, code]) => { const li = document.createElement("li"); li.append(el("span", "", display(area)), document.createTextNode(code)); return li; }));
  show("problems-empty", problems.length === 0);
  if ($("system-detail").open) text("system-json", JSON.stringify(s, null, 2));
}

// refresh loop --------------------------------------------------------------------------------
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
    delete $("status").dataset.stale;
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
let resizeTimer = null;
window.addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => { if (page === "today" && charts) drawCharts(); if (page === "record") refresh(true); }, 150); });
if (page === "today") { refreshCharts(); setInterval(refreshCharts, 60000); }
$("prev").onclick = () => { offset = Math.max(0, offset - 100); refresh(true); };
$("next").onclick = () => { offset += 100; refresh(true); };
$("system-detail").addEventListener("toggle", () => refresh(true));
document.addEventListener("visibilitychange", () => { if (document.hidden) pending?.abort(); else refresh(true); });
setInterval(() => {
  const now = new Date().toISOString();
  text("clock", `${ny(now)} NY · ${uk(now)} UK`);
  if (lastRefresh && Date.now() - Date.parse(lastRefresh) > (REFRESH_SECONDS * 3 + 5) * 1000) { text("last-refresh", `Stale · last refresh ${uk(lastRefresh)} UK`); $("status").dataset.stale = "1"; }
}, 1000);
setInterval(() => refresh(), REFRESH_SECONDS * 1000);
refresh(true);
