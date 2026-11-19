// Viewport (above-the-fold) captures for visual QA: node scripts/viewport-shots.mjs <outDir> [base]
import { chromium } from "@playwright/test";
const out = process.argv[2]; const base = process.argv[3] ?? "http://localhost:3000";
const shots = [
  ["desk-home", "/", 1440, 900, "text=Live pre-draft forecast"],
  ["desk-report", "/prospects/qb_226?mode=post_draft", 1440, 900, "#draft"],
  ["mob-home", "/", 390, 844, "h1"],
  ["tab-backtest", "/backtesting?id=qb_227", 820, 1180, "[data-testid=classification]"],
];
const b = await chromium.launch();
for (const [n, p, w, h, ready] of shots) {
  const page = await b.newPage({ viewport: { width: w, height: h } });
  await page.goto(base + p); await page.waitForSelector(ready, { timeout: 60000 }); await page.waitForTimeout(2500);
  await page.screenshot({ path: `${out}/${n}.png` }); await page.close(); console.log("saved", n);
}
await b.close();
