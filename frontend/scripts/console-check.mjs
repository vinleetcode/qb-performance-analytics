// Load pages and print browser console errors/warnings (dev aid).
import { chromium } from "@playwright/test";
const base = process.argv[2] ?? "http://localhost:3000";
const paths = ["/", "/prospects/qb_319", "/prospects/qb_226?mode=post_draft", "/backtesting?id=qb_227", "/methodology", "/prospects?q=allen"];
const browser = await chromium.launch();
const page = await browser.newPage();
page.on("console", (m) => { if (["error", "warning"].includes(m.type())) console.log(`[${m.type()}] ${page.url()} :: ${m.text().slice(0, 400)}`); });
page.on("pageerror", (e) => console.log(`[pageerror] ${page.url()} :: ${e.message}`));
for (const p of paths) { await page.goto(base + p, { waitUntil: "networkidle" }); await page.waitForTimeout(1500); }
await browser.close();
