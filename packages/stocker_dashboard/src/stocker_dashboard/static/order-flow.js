"use strict";
// Bounded summaries only. No transport or broker interface; stable shell/zoom/scroll.
const First4Flow = (() => {
  const fmt = (x) => typeof x === "number" && Number.isFinite(x)
    ? x.toLocaleString("en-US", {maximumFractionDigits: 1}) : "Unavailable";
  const pct = (x) => typeof x === "number" ? `${(100 * x).toFixed(1)}%` : "Unavailable";
  const set = (root, key, value) => {
    const el = root.querySelector(`[data-flow="${key}"]`);
    if (el && el.textContent !== value) el.textContent = value;
  };
  const at = (v) => v ? new Date(v).toISOString().slice(11, 19) + " UTC" : "unavailable";
  function cardShell() {
    return `<section class="flow-card" aria-label="Estimated order flow">
      <h4>Estimated share volume</h4><p data-flow="state"></p>
      <div class="flow-stack" role="img"><span class="flow-buy"></span><span class="flow-sell"></span><span class="flow-unknown"></span></div>
      <small data-flow="volumes"></small><dl><div><dt>Last minute Δ / shares</dt><dd data-flow="minute"></dd></div>
      <div><dt>Rolling 5m Δ / shares</dt><dd data-flow="five"></dd></div>
      <div><dt>Classified coverage / 5m</dt><dd data-flow="coverage"></dd></div></dl>
      <small data-flow="feed"></small><p data-flow="capture"></p><small data-flow="gap" class="flow-warning"></small>
      <p class="muted">Estimated direction, not a trading signal.</p></section>`;
  }
  function renderCard(card, flow) {
    const f = flow || {state: "DISABLED", feed_mode: "UNAVAILABLE"};
    set(card, "state", `${f.state}${f.reason ? " · " + f.reason : ""}`);
    const totals = f.rolling_5m;
    const values = [totals?.buy_est_volume, totals?.sell_est_volume, totals?.unknown_volume];
    const total = totals?.eligible_observed_volume || 0;
    card.querySelectorAll(".flow-stack span").forEach((bar, index) => {
      const width = `${total ? 100 * values[index] / total : 0}%`;
      if (bar.style.width !== width) bar.style.width = width;
    });
    const label = `5m: buy est ${fmt(values[0])} / sell est ${fmt(values[1])} / unknown ${fmt(values[2])} shares`;
    card.querySelector(".flow-stack").setAttribute("aria-label", label);
    set(card, "volumes", label + ` · excluded ${fmt(totals?.excluded_volume)}`);
    set(card, "minute", fmt(f.last_completed_minute?.volume_delta));
    set(card, "five", fmt(totals?.volume_delta));
    set(card, "coverage", pct(totals?.classified_volume_fraction));
    set(card, "feed", `${f.feed_mode} · quote age ${fmt(f.quote_age_ms)} ms · trade age ${fmt(f.trade_age_ms)} ms`);
    set(card, "capture", `Capture ${at(f.first_received_at)} · ${f.observation_state || "FEED UNAVAILABLE"}`);
    set(card, "gap", f.state === "DISABLED" ? "Observation disabled" :
      `${f.coverage_warning || "Coverage unavailable"} Pre-capture ${fmt(f.pre_capture_gap_seconds)} s · dropped ${f.dropped_events || 0} · ${totals?.partial_coverage ? "PARTIAL 5m WINDOW" : ""}`);
  }
  function detailShell() {
    return `<section id="flow-detail"><h3>Underlying order-flow observations</h3>
      <p data-flow="detail-state"></p><p data-flow="cumulative-label"></p>
      <label>Chart zoom <input id="flow-zoom" type="range" min="1" max="4" step="0.5" value="1"></label>
      <div class="flow-chart-scroll" tabindex="0" aria-label="Time aligned order-flow charts">
        <div class="flow-charts"><svg data-chart="price" aria-label="Underlying price USD"></svg>
        <svg data-chart="delta" aria-label="Estimated volume delta and unknown shares"></svg>
        <svg data-chart="cumulative" aria-label="Cumulative estimated volume delta shares"></svg></div></div>
      <p class="muted">Receipt time / UTC · green = estimated buy, orange = estimated sell, grey = unknown volume.
      Breaks and shaded columns mark gaps or unverified coverage. Each capture segment starts its own cumulative total.
      Rolling summaries use the last 1 / 5 completed receipt minutes. Quote sizes are displayed liquidity, never executed volume.</p></section>`;
  }
  let chartData = null;
  function renderDetail(flow, changed) {
    const root = document.querySelector("#flow-detail");
    if (!root) return;
    const f = flow || {state: "DISABLED", feed_mode: "UNAVAILABLE", bars: []};
    chartData = f;
    set(root, "detail-state", `${f.state} · ${f.feed_mode} · ${f.reason || f.coverage_warning || ""}`);
    set(root, "cumulative-label", `Cumulative estimated delta since ${at(f.first_received_at)} (current capture segment). Classified coverage ${pct(f.totals?.classified_volume_fraction)}.`);
    const zoom = document.querySelector("#flow-zoom");
    if (changed) zoom.value = "1";
    zoom.oninput = () => draw(chartData);
    draw(f);
  }
  function draw(flow) {
    const bars = flow.bars || [];
    const container = document.querySelector(".flow-charts");
    const zoom = Number(document.querySelector("#flow-zoom").value);
    container.style.width = `${100 * zoom}%`;
    const width = Math.max(760, container.clientWidth), height = 180, left = 90, right = 30, top = 26, bottom = 35;
    const times = bars.map(b => Date.parse(b.minute));
    const start = times.length ? Math.min(...times) : 0;
    const end = times.length ? Math.max(...times) + 60000 : 60000;
    const x = t => left + (t - start) / (end - start) * (width - left - right);
    const gaps = (flow.captures || []).flatMap(c => (c.gaps || []).map(g => Date.parse(g.at)));
    const columns = bars.map((b, i) => b.has_gap || i > 0 && (b.capture_id !== bars[i-1].capture_id || times[i] - times[i-1] > 60000) || gaps.some(g => times[i] <= g && g < times[i] + 60000));
    const ns = "http://www.w3.org/2000/svg";
    const make = (tag, attrs, text) => {
      const element = document.createElementNS(ns, tag);
      for (const [k, v] of Object.entries(attrs)) element.setAttribute(k, String(v));
      if (text !== undefined) element.textContent = text;
      return element;
    };
    for (const [kind, key, label] of [["price", "price", "Underlying / USD"], ["delta", "volume_delta", "Estimated volume delta / shares"], ["cumulative", "cumulative_volume_delta", "Cumulative estimated delta / shares"]]) {
      const svg = container.querySelector(`[data-chart="${kind}"]`);
      svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
      const items = bars.map(b => b[key]).filter(v => typeof v === "number" && Number.isFinite(v));
      const extent = kind === "delta" ? items.concat(bars.map(b => b.unknown_volume || 0)) : items;
      let low = kind === "price" && extent.length ? Math.min(...extent) : Math.min(0, ...extent);
      let high = Math.max(0, ...extent);
      if (high === low) { high += 1; if (kind === "price") low -= 1; }
      const y = v => top + (high - v) / (high - low) * (height - top - bottom);
      const layer = make("g", {});
      layer.append(make("text", {x: left, y: 15, class: "flow-axis"}, label));
      for (const value of [low, (high+low)/2, high]) {
        layer.append(make("line", {x1: left, x2: width-right, y1: y(value), y2: y(value), class: "flow-grid"}));
        layer.append(make("text", {x: left-8, y: y(value)+4, "text-anchor": "end", class: "flow-axis"}, fmt(value)));
      }
      for (let i=0; i<bars.length; i++) {
        if (columns[i]) layer.append(make("rect", {x:x(times[i]), y:top, width:Math.max(2,x(times[i]+60000)-x(times[i])), height:height-top-bottom, class:"flow-gap"}));
        const value = bars[i][key];
        if (typeof value !== "number") continue;
        if (kind === "delta") {
          const barWidth = Math.max(1, (width-left-right) * 60000/(end-start) * 0.7);
          for (const [v, cls, scale] of [[bars[i].unknown_volume || 0,"flow-unknown-bar",1], [value,value>=0?"flow-buy-bar":"flow-sell-bar",0.6]]) {
            const rect = make("rect", {x:x(times[i]), y:Math.min(y(0),y(v)), width:barWidth*scale, height:Math.max(0.5,Math.abs(y(v)-y(0))), class:cls});
            rect.append(make("title", {}, `${at(bars[i].minute)} · delta ${fmt(value)} · unknown ${fmt(bars[i].unknown_volume)} · classified ${pct(bars[i].classified_volume_fraction)}`));
            layer.append(rect);
          }
        } else {
          if (i > 0 && !columns[i] && !columns[i-1] && typeof bars[i-1][key] === "number") layer.append(make("line", {x1:x(times[i-1]), y1:y(bars[i-1][key]), x2:x(times[i]), y2:y(value), class:"flow-line"}));
          layer.append(make("circle", {cx:x(times[i]), cy:y(value), r:2, class:"flow-point"}));
        }
      }
      for (let i=0;i<=4;i++) {
        const t = start + (end-start)*i/4;
        layer.append(make("text", {x:x(t), y:height-8, "text-anchor":"middle", class:"flow-axis"}, bars.length ? new Date(t).toISOString().slice(11,16) : "No observed prints"));
      }
      svg.replaceChildren(layer); // Only bounded chart content, never the scroll/zoom shell.
    }
  }
  return {cardShell, renderCard, detailShell, renderDetail};
})();
