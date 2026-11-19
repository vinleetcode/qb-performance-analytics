// Screenshot one element at full resolution: node scripts/element-shot.mjs <url> <selector> <out.png> [width]
import { chromium } from "@playwright/test";
const [url, sel, out, width = "1440"] = process.argv.slice(2);
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: Number(width), height: 1000 }, deviceScaleFactor: 2 });
await page.goto(url, { waitUntil: "domcontentloaded" });
await page.waitForSelector(sel, { timeout: 60000 });
await page.waitForTimeout(1500);
await page.locator(sel).first().screenshot({ path: out });
await browser.close();
