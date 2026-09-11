// Browser regressions. All requests use fixtures; no broker is contacted.
const { chromium } = require("playwright");
const assert = require("node:assert/strict");
const fs = require("node:fs/promises");
const path = require("node:path");

(async () => {
  const browser = await chromium.launch({ headless: true, channel: process.env.PLAYWRIGHT_CHANNEL });
  try {
    const page = await browser.newPage();
    const errors = [];
    page.on("pageerror", error => errors.push(error.message));
    let symbol = "FIRST", fail = false, tradeSymbol = "TRADE_A", connected = false;
    const staticPath = path.join(__dirname, "../packages/stocker_dashboard/src/stocker_dashboard/static");
    await page.route("http://stocker.test/**", async route => {
      const url = new URL(route.request().url());
      if (url.pathname === "/api/overview") return route.fulfill({ json: { environments: {}, system: "READY", active_runs: 1, open_positions: 0, as_of: new Date().toISOString() } });
      if (url.pathname === "/api/runs") return route.fulfill({ json: [{run_id: "test", enabled: true}] });
      if (url.pathname === "/api/settings") return route.fulfill({ json: {
        runs: [{run_id: "test", strategy: "Session HARD"}], universes: [], strategies: [],
        broker: [{environment: "PAPER", connected}],
        broker_configuration: [{environment: "PAPER", host: "127.0.0.1", port: 4002, client_id: 1,
          expected_account: "FAKE", connect_timeout_seconds: 5, request_timeout_seconds: 5, market_data_line_budget: 100}]
      }});
      if (url.pathname === "/api/trades") return fail
        ? route.fulfill({ status: 503, json: { detail: "Read unavailable; reference fixture" } })
        : route.fulfill({ json: { items: [{symbol: tradeSymbol, environment: "PAPER"}], total: 1,
          summary: {trades: 1, total_pnl: 5, currency: "USD"} } });
      if (url.pathname === "/api/candidates") return fail
        ? route.fulfill({ status: 503, json: { detail: "Read unavailable; reference fixture" } })
        : route.fulfill({ json: { items: [{ symbol, status: "READY" }], total: 1 } });
      const file = url.pathname.startsWith("/static/") ? path.basename(url.pathname) : "index.html";
      return route.fulfill({ body: await fs.readFile(path.join(staticPath, file)), contentType: file.endsWith(".js") ? "text/javascript" : file.endsWith(".css") ? "text/css" : "text/html" });
    });
    await page.goto("http://stocker.test/candidates");
    await page.getByText("FIRST", { exact: true }).waitFor();
    await page.evaluate(() => clearTimeout(timer));
    const input = page.locator("#candidate-checkpoint");
    await input.focus();
    await input.evaluate(node => { node.value = "2026-09-11T14:00"; });
    symbol = "SECOND";
    await page.evaluate(() => refreshCurrentPage());
    assert.equal(await input.inputValue(), "2026-09-11T14:00");
    assert.equal(await input.evaluate(node => document.activeElement === node), true);
    assert.match(await page.locator("#candidate-results").innerText(), /SECOND/);
    const stalePanel = await page.locator("#candidate-results").innerHTML();
    await page.evaluate(() => refreshHeader());
    assert.equal(await page.locator("#candidate-results").innerHTML(), stalePanel,
      "a successful header read must not make stale panel data fresh");
    fail = true;
    await page.evaluate(() => refreshCurrentPage());
    assert.match(await page.locator("#candidate-results").innerText(), /STALE.*Received/s);
    assert.match(await page.locator("#candidate-results").innerText(), /SECOND/);
    fail = false;
    symbol = "RECOVERED";
    await page.evaluate(() => refreshCurrentPage());
    assert.match(await page.locator("#candidate-results").innerText(), /RECOVERED/);
    assert.doesNotMatch(await page.locator("#candidate-results").innerText(), /STALE/);

    // Deterministic timers exercise cancellation, recovery, and uncertain mutations.
    await page.clock.install();
    await page.evaluate(() => {
      clearTimeout(timer);
      window.originalFetch = window.fetch;
      window.mutationCalls = 0;
      window.fetch = (_url, options) => new Promise((_resolve, reject) => {
        if (options.method === "POST") ++window.mutationCalls;
        options.signal.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")));
      });
      window.readResult = api("/hung").catch(error => error.message);
    });
    await page.clock.runFor(8001);
    assert.match(await page.evaluate(() => window.readResult), /Read timed out/);
    await page.evaluate(() => { window.mutationResult = api("/control", { method: "POST" }).catch(error => error.message); });
    await page.clock.runFor(30001);
    assert.match(await page.evaluate(() => window.mutationResult), /outcome unknown/);
    assert.equal(await page.evaluate(() => window.mutationCalls), 1);
    await page.evaluate(() => { window.fetch = window.originalFetch; });
    assert.equal((await page.evaluate(() => api("/api/runs")))[0].run_id, "test");

    // Exercise the actual polling cycle, not only the request helper.
    await page.evaluate(() => {
      clearTimeout(timer);
      window.fetch = (url, options) => {
        if (!String(url).startsWith("/api/candidates")) return window.originalFetch(url, options);
        window.panelReadStarted = true;
        return new Promise((_resolve, reject) => options.signal.addEventListener("abort",
          () => reject(new DOMException("Aborted", "AbortError"))));
      };
      window.hungRefresh = refreshCurrentPage();
    });
    await page.waitForFunction(() => window.panelReadStarted === true);
    await page.clock.runFor(8001);
    await page.evaluate(() => window.hungRefresh);
    assert.equal(await page.evaluate(() => refreshingPage), false);
    assert.equal(await page.evaluate(() => readControllers.size), 0);
    assert.match(await page.locator("#candidate-results").innerText(), /STALE.*RECOVERED/s);
    symbol = "POLL_RECOVERED";
    await page.evaluate(() => { window.fetch = window.originalFetch; });
    await page.clock.runFor(10001);
    await page.getByText("POLL_RECOVERED", {exact: true}).waitFor();
    assert.doesNotMatch(await page.locator("#candidate-results").innerText(), /STALE/);
    assert.equal(await page.evaluate(() => window.mutationCalls), 1);
    await page.evaluate(() => clearTimeout(timer));
    assert.equal(await page.evaluate(() => riskFraction("0.1")), 0.001);
    for (const value of ["", "NaN", "0", "101", "-1"]) {
      assert.equal(await page.evaluate(value => { try { riskFraction(value); return false; } catch (_) { return true; } }, value), true);
    }

    // A late table result must not overwrite a newer render.
    await page.evaluate(() => {
      clearTimeout(timer);
      window.fetch = async (url, options) => {
        if (String(url).startsWith("/api/candidates")) return new Promise(resolve => {
          window.releaseOld = () => resolve(new Response(JSON.stringify({items: [{symbol: "OLD"}], total: 1}), {headers: {"Content-Type": "application/json"}}));
        });
        return window.originalFetch(url, options);
      };
      window.oldRequest = candidatesPage(true);
    });
    await page.waitForFunction(() => typeof window.releaseOld === "function");
    await page.evaluate(() => { ++pageRevision; window.fetch = window.originalFetch; });
    symbol = "NEWEST";
    await page.evaluate(() => candidatesPage(true));
    await page.evaluate(async () => { window.releaseOld(); await window.oldRequest; });
    assert.match(await page.locator("#candidate-results").innerText(), /NEWEST/);
    assert.doesNotMatch(await page.locator("#candidate-results").innerText(), /OLD/);
    await page.goto("http://stocker.test/trades");
    await page.getByText("TRADE_A", {exact: true}).waitFor();
    const tradeInput = page.locator("#trade-symbol");
    await tradeInput.focus();
    await tradeInput.evaluate(node => { node.value = "UNSAVED"; });
    tradeSymbol = "TRADE_B";
    await page.evaluate(() => refreshCurrentPage());
    assert.match(await page.locator("#trade-results").innerText(), /TRADE_B/);
    assert.equal(await tradeInput.inputValue(), "UNSAVED");
    assert.equal(await tradeInput.evaluate(node => document.activeElement === node), true);
    fail = true;
    await page.evaluate(() => refreshCurrentPage());
    assert.match(await page.locator("#trade-results").innerText(), /STALE.*TRADE_B/s);
    fail = false;

    await page.goto("http://stocker.test/settings");
    await page.getByText("DISCONNECTED", {exact: false}).waitFor();
    const host = page.locator('input[name="host"]');
    await host.focus();
    await host.evaluate(node => { node.value = "unsaved.local"; });
    connected = true;
    await page.evaluate(() => refreshCurrentPage());
    assert.match(await page.locator("#broker-runtime-PAPER").innerText(), /CONNECTED/);
    assert.doesNotMatch(await page.locator("#broker-runtime-PAPER").innerText(), /DISCONNECTED/);
    assert.equal(await host.inputValue(), "unsaved.local");
    assert.equal(await host.evaluate(node => document.activeElement === node), true);
    assert.deepEqual(errors, []);
    console.log("PASS: panel freshness, form/focus preservation, timeout recovery, uncertain controls, percent conversion, response ordering");
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
