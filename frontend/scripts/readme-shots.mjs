// Capture the README screenshots from a running app + API: node scripts/readme-shots.mjs [outDir] [baseUrl]
// The images show real model output for real players (see the licensing note in the README before publishing).
import { chromium } from "@playwright/test";

const out = process.argv[2] ?? "docs/screenshots";
const base = process.argv[3] ?? "http://localhost:3000";
const shots = [
  { name: "landing", path: "/", ready: "text=The 2026 quarterback class", w: 1440, h: 900 },
  { name: "report-production", path: "/prospects/qb_226?mode=post_draft", ready: "#production", w: 1440, h: 1100, scrollTo: "#production" },
  { name: "report-header", path: "/prospects/qb_226?mode=post_draft", ready: "#draft", w: 1440, h: 1000 },
  { name: "backtest", path: "/backtesting?id=qb_227", ready: "[data-testid=classification]", w: 1440, h: 1100, scrollTo: "section[aria-label='Selected backtest']" },
  { name: "methodology", path: "/methodology", ready: "#draft", w: 1440, h: 1000, scrollTo: "#draft" },
  { name: "mobile-report", path: "/prospects/qb_226?mode=post_draft", ready: "#starts", w: 390, h: 844, scrollTo: "#starts" },
];
const browser = await chromium.launch();
for (const s of shots) {
  const page = await browser.newPage({ viewport: { width: s.w, height: s.h }, deviceScaleFactor: 2 });
  await page.goto(base + s.path, { waitUntil: "domcontentloaded" });
  await page.waitForSelector(s.ready, { timeout: 60000 });
  await page.waitForTimeout(1500);
  if (s.scrollTo) await page.locator(s.scrollTo).first().evaluate((el) => window.scrollTo(0, el.getBoundingClientRect().top + window.scrollY - 60));
  await page.waitForTimeout(500);
  await page.screenshot({ path: `${out}/${s.name}.png` });
  await page.close();
  console.log("saved", s.name);
}
await browser.close();
