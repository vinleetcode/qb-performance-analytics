// Render pages against the live dev server + API and save screenshots for visual review.
import { chromium } from "@playwright/test";

const out = process.argv[2];
const base = process.argv[3] ?? "http://localhost:3000";
const pages = [
  ["landing", "/", "text=The 2026 quarterback class"],
  ["prospects", "/prospects?q=allen", "text=quarterbacks"],
  ["report_pre", "/prospects/qb_319", "#production"],
  ["report_post", "/prospects/qb_226?mode=post_draft", "#production"],
  ["backtest", "/backtesting?id=qb_227", "[data-testid=classification]"],
  ["methodology", "/methodology", "#limits"],
];
const browser = await chromium.launch();
for (const [w, h, tag] of [[1440, 1000, "desktop"], [390, 844, "mobile"]]) {
  const ctx = await browser.newContext({ viewport: { width: w, height: h }, deviceScaleFactor: 1 });
  const page = await ctx.newPage();
  for (const [name, path, ready] of pages) {
    await page.goto(base + path, { waitUntil: "domcontentloaded" });
    try {
      await page.waitForSelector(ready, { timeout: 60000 });
    } catch {
      console.log("timeout", name, tag);
    }
    await page.waitForTimeout(1200);
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
    console.log(tag, name, "horizontal overflow px:", overflow);
    await page.screenshot({ path: `${out}/${tag}_${name}.png`, fullPage: true });
  }
  await ctx.close();
}
await browser.close();
