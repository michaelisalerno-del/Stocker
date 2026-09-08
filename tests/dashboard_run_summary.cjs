// All API traffic is mocked; no broker connection or order submission.
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
    let requests = 123;
    let auditDownloads = 0;
    const staticPath = path.join(__dirname, "../packages/stocker_dashboard/src/stocker_dashboard/static");
    await page.context().route("http://stocker.test/**", async route => {
      const url = new URL(route.request().url());
      assert.equal(route.request().method(), "GET");
      const json = value => route.fulfill({ json: value });
      if (url.pathname === "/api/overview") return json({ environments: {}, system: "READY", active_runs: 1, open_positions: 0, as_of: "2026-09-08T19:00:00Z" });
      if (url.pathname === "/api/runs/test") return json({
        run_id: "test", display_name: "US · Session HARD", environment: "PAPER", enabled: true,
        evaluation_status: "NO_MORE_CHECKPOINTS_TODAY", last_checkpoint: "2026-09-08T16:00:00Z",
        downloads: { scope: "ALL_RUNS", requests, pending: 4 },
        method_spec: { entry: { trigger_M: 0.2 } },
        funnel: [{ stage: "Stock eligibility", count: 6570 }, { stage: "Required data ready", count: 153 }],
      });
      if (url.pathname === "/api/runs/test/performance") return json({ history: [], currency: "USD", closed_trades: 0 });
      if (url.pathname === "/api/runs/test/provenance") {
        auditDownloads++;
        return route.fulfill({ json: { historical: "FULL_UNIVERSE_DATA" }, headers: { "Content-Disposition": 'attachment; filename="run-provenance.json"' } });
      }
      const file = url.pathname.startsWith("/static/") ? path.basename(url.pathname) : "index.html";
      return route.fulfill({ body: await fs.readFile(path.join(staticPath, file)), contentType: file.endsWith(".js") ? "text/javascript" : file.endsWith(".css") ? "text/css" : "text/html" });
    });
    await page.goto("http://stocker.test/runs?run=test");
    await page.getByText("No further method checkpoints today.", { exact: false }).waitFor();
    assert.match(await page.locator("main").innerText(), /123 history requests/);
    assert.equal(auditDownloads, 0);
    assert.equal(await page.locator("main").getByRole("row").count(), 0);
    assert(await page.getByRole("link", { name: "Browse data-ready stocks" }).getAttribute("href").then(href => href.includes("status=READY")));
    requests = 127;
    await page.evaluate(() => refreshCurrentPage());
    assert.match(await page.locator("main").innerText(), /127 history requests/);
    assert.equal(auditDownloads, 0);
    await page.getByText("Method specification", { exact: true }).click();
    assert.equal(await page.getByRole("link", { name: "Download full run audit record" }).getAttribute("download"), "run-provenance.json");
    // Chromium cancels native downloads for intercepted fictitious hosts. Exercise the same
    // explicit GET through fetch; the API suite verifies the attachment response header.
    const audit = await page.evaluate(async () => {
      const response = await fetch(document.querySelector("a[download]").href);
      return response.json();
    });
    assert.equal(audit.historical, "FULL_UNIVERSE_DATA");
    assert.equal(auditDownloads, 1);
    assert(!await page.locator("main").innerText().then(text => text.includes("FULL_UNIVERSE_DATA")));
    assert.deepEqual(errors, []);
    console.log("PASS: compact summary, live download refresh, checkpoint status, on-demand audit, GET-only");
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
