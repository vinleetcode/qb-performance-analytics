/**
 * Real integration: these tests drive the Next.js app against the live FastAPI backend and its real models.
 * Expected values (names, picks, classifications) come from the backend's serving bundle, not from fixtures.
 */
import { expect, test, type Page } from "@playwright/test";
import AxeBuilder from "./axe";

const API = process.env.E2E_API_URL ?? "http://localhost:8000";

async function noHorizontalOverflow(page: Page) {
  const extra = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  expect(extra).toBeLessThanOrEqual(0);
}

test.beforeAll(async ({ request }) => {
  const r = await request.get(`${API}/api/health`);
  expect(r.ok(), "start the backend: uvicorn api.main:app --port 8000").toBeTruthy();
});

test("landing shows the newest class and a live forecast", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { level: 1 })).toContainText("From college production to NFL projection.");
  await expect(page.getByRole("link", { name: "Explore prospects" })).toBeVisible();
  await expect(page.getByRole("link", { name: "See historical accuracy" })).toBeVisible();
  await expect(page.getByText("Fernando Mendoza").first()).toBeVisible();
  await expect(page.getByText("Pick 1", { exact: true })).toBeVisible();
  await expect(page.getByRole("img", { name: /Passing yards, NFL seasons 1 to 5: expected/ })).toBeVisible({ timeout: 60_000 });
});

test("search by name, open a report by stable id", async ({ page }) => {
  await page.goto("/");
  const box = page.getByRole("combobox");
  await box.fill("josh allen");
  const opt = page.getByRole("option").filter({ hasText: "Wyoming" });
  await expect(opt).toBeVisible();
  await expect(page.getByRole("option").first()).toContainText(/Josh Allen/);
  await opt.click();
  await expect(page).toHaveURL(/\/prospects\/qb_226/);
  await expect(page.getByRole("heading", { level: 1 })).toHaveText("Josh Allen");
});

test("pre-draft hides the pick; post-draft shows it and changes the forecast", async ({ page }) => {
  await page.goto("/prospects/qb_226");
  await expect(page.locator("#production")).toBeVisible({ timeout: 60_000 });
  await expect(page.getByText("Actual draft result hidden in the pre-draft view")).toBeVisible();
  const preYards = await page.getByRole("img", { name: /^Passing yards, NFL seasons 1 to 5/ }).getAttribute("aria-label");
  await page.getByRole("radio", { name: "Post-draft" }).click();
  await expect(page).toHaveURL(/mode=post_draft/);
  await expect(page.getByText("Drafted round 1, pick 7")).toBeVisible();
  await expect(page.getByRole("img", { name: /Overall pick, if drafted:.*actual pick #7/ })).toBeVisible({ timeout: 60_000 });
  const postYards = await page.getByRole("img", { name: /^Passing yards, NFL seasons 1 to 5/ }).getAttribute("aria-label");
  expect(postYards).not.toEqual(preYards);
});

test("nested starting probabilities and intervals come from the API", async ({ page, request }) => {
  const api = await (await request.get(`${API}/api/prospects/qb_226/forecast?mode=pre_draft`)).json();
  const s = api.career.start_probabilities;
  expect(s.p_64_plus).toBeLessThanOrEqual(s.p_32_plus);
  expect(s.p_32_plus).toBeLessThanOrEqual(s.p_16_plus);
  await page.goto("/prospects/qb_226");
  const starts = page.locator("#starts");
  await expect(starts.getByText(`${Math.round(100 * s.p_32_plus)}%`, { exact: true })).toBeVisible({ timeout: 60_000 });
  const py = api.career.five_year_totals.find((x: { stat: string }) => x.stat === "pass_yards");
  const table = page.getByRole("table", { name: /All ten projections/ });
  await expect(table.getByText(Math.round(py.expected).toLocaleString("en-US"), { exact: true })).toBeVisible();
});

test("unsupported outputs are explained, not zeroed", async ({ page }) => {
  await page.goto("/prospects/qb_319"); // 2026 class: final season not verifiable yet
  await expect(page.getByText("Probability of being drafted is not available for this player")).toBeVisible({ timeout: 60_000 });
  await page.goto("/prospects/cfb_4606194_2024?mode=post_draft"); // undrafted: no real pick
  await expect(page.getByText(/has no recorded NFL draft slot/)).toBeVisible({ timeout: 60_000 });
  await page.getByLabel("Hypothetical overall pick (1 to 262)").fill("150");
  await page.getByRole("button", { name: "Forecast at this pick" }).click();
  await expect(page.getByText(/Hypothetical: these forecasts assume a draft slot you chose/)).toBeVisible({ timeout: 60_000 });
});

test("comparisons link to other reports", async ({ page }) => {
  await page.goto("/prospects/qb_226");
  const comps = page.locator("#comparisons");
  await expect(comps.getByRole("listitem")).toHaveCount(5, { timeout: 60_000 });
  await expect(comps.getByText(/not a probability/i).or(comps.getByText(/not a forecast/i)).first()).toBeAttached();
  const first = comps.getByRole("link").first();
  const href = await first.getAttribute("href");
  expect(href).toMatch(/^\/prospects\/qb_\d+$/);
});

test("backtest classifications are labelled honestly", async ({ page }) => {
  await page.goto("/backtesting?id=qb_227");
  await expect(page.getByTestId("classification")).toContainText("Genuine out-of-sample backtest", { timeout: 60_000 });
  await expect(page.getByRole("table", { name: /Predicted vs actual, NFL seasons 1 to 5/ })).toBeVisible();
  await page.goto("/backtesting?id=qb_107");
  await expect(page.getByTestId("classification")).toContainText("In-sample illustration, not a backtest");
  await expect(page.getByText(/trained on this player/)).toBeVisible();
  await page.goto("/backtesting?id=qb_281");
  await expect(page.getByTestId("classification")).toContainText("NFL window still open");
  await expect(page.getByText("Not yet known").first()).toBeVisible();
  await expect(page.getByRole("heading", { name: "Accuracy across all backtests" })).toBeVisible();
  await expect(page.getByText(/averages over all 107 drafted quarterbacks/)).toBeVisible();
});

test("methodology explains the Combine finding with backend numbers", async ({ page }) => {
  await page.goto("/methodology");
  await expect(page.getByText(/Adding only whether a player was invited/)).toBeVisible({ timeout: 60_000 });
  await expect(page.getByText(/Why the draft slot matters so much/)).toBeVisible();
  await expect(page.getByRole("heading", { name: "Limitations" })).toBeVisible();
});

test("backend outage shows an error, never invented numbers", async ({ page }) => {
  test.setTimeout(150_000);
  await page.route(`${API}/**`, (r) => r.abort());
  await page.goto("/prospects/qb_226");
  // while it retries (a sleeping free-tier backend takes about a minute to wake) the page says so honestly
  await expect(page.getByText(/free hosting tier and sleeps when idle/).first()).toBeVisible({ timeout: 15_000 });
  await expect(page.getByRole("alert").filter({ hasText: "Forecasting service unavailable" }).first()).toBeVisible({ timeout: 120_000 });
  await expect(page.locator("#production")).toHaveCount(0);
});

test("homepage stays usable while the backend is down", async ({ page }) => {
  await page.route(`${API}/**`, (r) => r.abort());
  await page.goto("/");
  await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
  await expect(page.getByRole("link", { name: /methodology/i }).first()).toBeVisible();
  await expect(page.locator("img").first()).toBeVisible();
});

test("keyboard: skip link and focusable search", async ({ page }) => {
  await page.goto("/");
  await page.keyboard.press("Tab");
  await expect(page.getByRole("link", { name: "Skip to content" })).toBeFocused();
  await page.getByRole("combobox").focus();
  await expect(page.getByRole("combobox")).toBeFocused();
});

for (const path of ["/", "/prospects?q=allen", "/prospects/qb_226?mode=post_draft", "/backtesting?id=qb_227", "/methodology", "/case-study", "/credits"]) {
  test(`no serious accessibility violations on ${path}`, async ({ page }) => {
    await page.goto(path);
    await page.waitForLoadState("networkidle");
    await page.waitForTimeout(1500);
    const v = await new AxeBuilder(page).analyze();
    const serious = v.filter((x) => x.impact === "serious" || x.impact === "critical");
    expect(serious.map((x) => `${x.id} (${x.nodes.length})`)).toEqual([]);
  });
}

for (const path of ["/", "/prospects?q=allen", "/prospects/qb_319", "/prospects/qb_226?mode=post_draft", "/backtesting?id=qb_227", "/methodology", "/case-study", "/credits"]) {
  test(`@mobile no horizontal overflow on ${path}`, async ({ page }) => {
    await page.goto(path);
    await page.waitForLoadState("networkidle");
    await page.waitForTimeout(1000);
    await noHorizontalOverflow(page);
  });
}

const ALL = ["/", "/prospects", "/prospects/qb_226?mode=post_draft", "/prospects/cfb_4606194_2024", "/backtesting?id=qb_107", "/methodology", "/case-study", "/credits"];

test("every page loads its photos and logs no console errors", async ({ page }) => {
  const errors: string[] = [];
  page.on("console", (m) => m.type() === "error" && errors.push(`${page.url()}: ${m.text()}`));
  page.on("pageerror", (e) => errors.push(`${page.url()}: ${e.message}`));
  for (const path of ALL) {
    await page.goto(path);
    await page.waitForLoadState("networkidle");
    await page.waitForTimeout(800);
    const broken = await page.evaluate(() =>
      Array.from(document.images).filter((i) => i.complete && i.naturalWidth === 0).map((i) => i.currentSrc || i.src),
    );
    expect(broken, path).toEqual([]);
  }
  expect(errors).toEqual([]);
});

test("case study and credits render with attribution", async ({ page }) => {
  await page.goto("/case-study");
  await expect(page.getByRole("heading", { name: "Case study: forecasting quarterback careers" })).toBeVisible();
  await expect(page.getByText(/smaller than the historical median before/)).toBeVisible({ timeout: 30_000 });
  await page.goto("/credits");
  await expect(page.getByText(/licensed CC BY 4.0/)).toBeVisible();
  await expect(page.getByRole("link", { name: "source", exact: true })).toHaveCount(6);
});

test("@mobile menu opens and navigates", async ({ page }) => {
  test.skip((page.viewportSize()?.width ?? 0) >= 768, "the menu button exists only below the md breakpoint");
  await page.goto("/");
  await page.getByRole("button", { name: "Open menu" }).click();
  await page.locator("#mobile-menu").getByRole("link", { name: "Backtesting" }).click();
  await expect(page).toHaveURL(/\/backtesting/);
  await expect(page.locator("#mobile-menu")).toHaveCount(0);
});

for (const path of ["/", "/prospects/qb_226", "/backtesting?id=qb_227", "/methodology"]) {
  test(`tablet (820 px) has no horizontal overflow on ${path}`, async ({ page }) => {
    await page.setViewportSize({ width: 820, height: 1180 });
    await page.goto(path);
    await page.waitForLoadState("networkidle");
    await page.waitForTimeout(800);
    await noHorizontalOverflow(page);
  });
}

/* ---------------------------------------------------------------- Phase 8: forecast presentation */

type Iv = { level: number; lower: number; upper: number };
type Stat = { stat: string; label: string; expected: number; intervals: Iv[]; measured_coverage?: { c80?: number } | null };
const n0 = (v: number) => Math.round(v).toLocaleString("en-US");
const at = (s: Stat, l: number) => s.intervals.find((i) => Math.abs(i.level - l) < 1e-9)!;
const HEADLINE = ["pass_yards", "pass_tds", "interceptions", "rush_yards", "games_started"];

test("headline cards, range chart and full table show exactly the API's values", async ({ page, request }) => {
  const api = await (await request.get(`${API}/api/prospects/qb_226/forecast?mode=post_draft`)).json();
  const totals: Stat[] = api.career.five_year_totals;
  const by = new Map(totals.map((s) => [s.stat, s]));
  await page.goto("/prospects/qb_226?mode=post_draft");
  const cards = page.getByRole("list", { name: "Headline forecast, NFL seasons 1 to 5" }).getByRole("listitem");
  await expect(cards).toHaveCount(5, { timeout: 60_000 });
  for (const [i, k] of HEADLINE.entries()) {
    const s = by.get(k)!;
    await expect(cards.nth(i)).toContainText(s.label);
    await expect(cards.nth(i).getByText(n0(s.expected), { exact: true })).toBeVisible();
    await expect(cards.nth(i)).toContainText(`${n0(at(s, 0.8).lower)} to ${n0(at(s, 0.8).upper)}`);
  }
  for (const k of HEADLINE) {
    const s = by.get(k)!;
    await page.locator("fieldset label").filter({ hasText: new RegExp(`^${s.label}$`) }).click();
    await expect(page.getByRole("radio", { name: s.label })).toBeChecked();
    const label = await page.getByRole("img", { name: /^Range chart for/ }).getAttribute("aria-label");
    expect(label).toContain(`expected ${n0(s.expected)}`);
    expect(label).toContain(`50% range ${n0(at(s, 0.5).lower)} to ${n0(at(s, 0.5).upper)}`);
    expect(label).toContain(`80% range ${n0(at(s, 0.8).lower)} to ${n0(at(s, 0.8).upper)}`);
  }
  const table = page.getByRole("table", { name: /All ten projections/ });
  await expect(table.getByRole("rowheader")).toHaveCount(totals.length);
  for (const s of totals) {
    const row = table.getByRole("row").filter({ has: page.getByRole("rowheader", { name: new RegExp(`${s.label}$`) }) });
    await expect(row).toContainText(n0(s.expected));
    await expect(row).toContainText(`${n0(at(s, 0.5).lower)} to ${n0(at(s, 0.5).upper)}`);
    await expect(row).toContainText(`${n0(at(s, 0.8).lower)} to ${n0(at(s, 0.8).upper)}`);
    const c = s.measured_coverage?.c80;
    if (c != null) {
      await expect(row).toContainText(`${Math.round(100 * c)}%`);
      if (Math.round(100 * c) < 75) await expect(row.getByText("Runs narrow")).toBeVisible(); // weak coverage is never hidden
      else await expect(row.getByText("Runs narrow")).toHaveCount(0);
    }
  }
});

test("50% ranges sit inside 80% ranges for a sample of prospects in both modes", async ({ request }) => {
  const ids: { id: string; drafted: boolean }[] = [];
  for (const year of [2008, 2014, 2019, 2023, 2026]) {
    const r = await (await request.get(`${API}/api/prospects?draft_year=${year}&limit=6`)).json();
    for (const p of r.items) ids.push({ id: p.player_id, drafted: p.overall_pick != null });
  }
  expect(ids.length).toBeGreaterThan(15);
  let checked = 0;
  for (const { id, drafted } of ids) {
    for (const mode of drafted ? ["pre_draft", "post_draft"] : ["pre_draft"]) {
      const res = await request.get(`${API}/api/prospects/${id}/forecast?mode=${mode}`);
      expect([200, 422], `${id} ${mode}: HTTP ${res.status()} (429 means the run hit the API rate limit)`).toContain(res.status());
      const f = await res.json();
      if (f.career?.status !== "available") continue;
      for (const s of f.career.five_year_totals as Stat[]) {
        const a = at(s, 0.5), b = at(s, 0.8);
        expect(a.lower, `${id} ${mode} ${s.stat}`).toBeGreaterThanOrEqual(b.lower - 1e-6);
        expect(a.upper, `${id} ${mode} ${s.stat}`).toBeLessThanOrEqual(b.upper + 1e-6);
        expect(b.lower).toBeLessThanOrEqual(b.upper);
        checked++;
      }
    }
  }
  expect(checked).toBeGreaterThan(150);
});

test("range explanations open with the mouse and the keyboard", async ({ page }) => {
  await page.goto("/prospects/qb_226");
  const table = page.getByRole("table", { name: /All ten projections/ });
  await expect(table).toBeVisible({ timeout: 60_000 });
  await table.getByRole("button", { name: "About 80% range" }).hover();
  await expect(page.getByText("A wider range, meant to contain about 80% of comparable outcomes.")).toBeVisible();
  await page.mouse.move(0, 0);
  await expect(page.getByText("A wider range, meant to contain about 80% of comparable outcomes.")).toHaveCount(0);
  await table.getByRole("button", { name: "About 80% coverage in backtests" }).focus();
  await page.keyboard.press("Enter");
  await expect(page.getByText(/share of historical outcomes that actually fell inside/)).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(page.getByText(/share of historical outcomes that actually fell inside/)).toHaveCount(0);
});

test("@mobile phones get projection cards and tooltips open on tap", async ({ page }) => {
  test.skip((page.viewportSize()?.width ?? 0) >= 768, "card layout is for phones");
  await page.goto("/prospects/qb_226?mode=post_draft");
  await expect(page.getByRole("list", { name: "Headline forecast, NFL seasons 1 to 5" })).toBeVisible({ timeout: 60_000 });
  await expect(page.getByRole("table", { name: /All ten projections/ })).toBeHidden();
  await expect(page.getByRole("region", { name: "Rushing" })).toBeVisible();
  await page.getByRole("button", { name: "About 50% range" }).first().tap();
  await expect(page.getByText("A narrower range, meant to contain about half of comparable outcomes.")).toBeVisible();
  await noHorizontalOverflow(page);
});
