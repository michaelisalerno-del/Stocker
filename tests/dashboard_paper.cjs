/* The three pages against fixture JSON dumped from the offline paper day (tests/fixtures). No Saxo calls. */
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const http = require("node:http");
const {chromium} = require("playwright");
const root = path.resolve("packages/stocker_dashboard/src/stocker_dashboard/static");
const fixture = JSON.parse(fs.readFileSync(path.resolve("tests/fixtures/paper_dashboard.json"), "utf8"));
const calls = [];
const server = http.createServer((req, res) => {
  const url = new URL(req.url, "http://localhost");
  if (url.pathname.startsWith("/api/")) {
    calls.push(url.pathname);
    res.setHeader("Content-Type", "application/json");
    return res.end(JSON.stringify(fixture[url.pathname] || {}));
  }
  const file = url.pathname.startsWith("/static/") ? path.basename(url.pathname) : "index.html";
  res.setHeader("Content-Type", file.endsWith(".js") ? "text/javascript" : file.endsWith(".css") ? "text/css" : file.endsWith(".svg") ? "image/svg+xml" : "text/html");
  res.end(fs.readFileSync(path.join(root, file)));
});
(async () => {
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const browser = await chromium.launch({headless: true});
  try {
    const base = `http://127.0.0.1:${server.address().port}`;
    for (const viewport of [{width: 1280, height: 900}, {width: 390, height: 844}]) {
      const page = await browser.newPage({viewport, timezoneId: "Europe/London"});
      const errors = [];
      page.on("pageerror", (e) => errors.push(e.message));

      await page.goto(base + "/");
      await page.waitForFunction(() => document.querySelectorAll("#cfd tr").length === 2);
      assert.equal(await page.locator("nav a[aria-current]").textContent(), "Today");
      assert.match(await page.locator("#sig-NAS100").textContent(), /^Long · candle 30018\.00 vs EMA12/);
      assert.match(await page.locator("#sig-US500").textContent(), /^Short/);
      assert.equal(await page.locator("#plan tr").count(), 10);
      assert.match(await page.locator("#plan tr").first().textContent(), /First candle · CFD trades.*09:35.*14:35.*Done/);
      assert.match(await page.locator("#cfd tr").nth(1).textContent(), /Stop 09:40 NY.*-\$204\.00/);
      assert.match(await page.locator("#options tr").first().textContent(), /Call 701.*\+\$59\.92/);
      assert.match(await page.locator("#crosses tr").first().textContent(), /10:30 NY · QQQ 701\.00 vs VWAP 700\.08.*Back below vwap.*\+\$65\.34/);
      assert.match(await page.locator("#crosses tr").nth(1).textContent(), /Missed entry · Qqq chart delayed/);
      assert.match(await page.locator("#sig-DIP").textContent(), /^Connors: buy · Turnaround Tuesday: buy · QQQ 710\.00 \(-1\.11% on the day\) · Fear rule: no reading$/);
      assert.equal(await page.locator("#play-NAS100").textContent(), "Waiting for the 09:35 signal");
      assert.match(await page.locator("#dips tr").first().textContent(), /Connors · in the book.*Rsi2 above 65 2026-10-19.*\+\$197\.34/);
      assert.match(await page.locator("#dips tr").nth(1).textContent(), /Turnaround Tuesday.*Open · sells on its rule/);
      assert.equal(await page.locator("#status").getAttribute("data-tone"), "ok");
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth), 0);

      await page.goto(base + "/record");
      await page.waitForFunction(() => document.querySelectorAll("#summary tr").length === 3);
      assert.equal(await page.locator("nav a[aria-current]").textContent(), "Record");
      assert.equal(await page.locator("#trades tr").count(), 3);
      assert.ok(await page.locator("#prev").isDisabled());
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth), 0);

      await page.goto(base + "/system");
      await page.waitForFunction(() => document.querySelector("#auth-state").textContent.includes("AUTHENTICATED"));
      assert.equal(await page.locator("#facts div").count(), 8);
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth), 0);
      assert.deepEqual(errors, []);
      await page.close();
    }
    // Each page asks only for its own data.
    assert.deepEqual([...new Set(calls)].sort(), ["/api/record", "/api/system", "/api/today"]);
    console.log("PASS: Today, Record and System render the paper day at desktop and phone width");
  } finally {
    await browser.close();
    await new Promise((resolve) => server.close(resolve));
  }
})().catch((error) => { console.error(error); process.exitCode = 1; });
