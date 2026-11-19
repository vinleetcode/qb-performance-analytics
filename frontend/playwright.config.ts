import { defineConfig, devices } from "@playwright/test";

/**
 * End-to-end tests against the REAL stack: the FastAPI backend must already be running on :8000
 * (`uvicorn api.main:app --port 8000` from nfl-qb-analytics/). The frontend is started automatically unless one is
 * already listening on :3000. Set E2E_BASE_URL to test a production build (`npm run build && npm start`).
 */
export default defineConfig({
  testDir: "./e2e",
  timeout: 90_000,
  expect: { timeout: 30_000 },
  fullyParallel: false,
  workers: 1,
  reporter: [["list"]],
  use: { baseURL: process.env.E2E_BASE_URL ?? "http://localhost:3000", trace: "off" },
  webServer: process.env.E2E_BASE_URL
    ? undefined
    : { command: "npm run dev", url: "http://localhost:3000", reuseExistingServer: true, timeout: 180_000 },
  projects: [
    { name: "desktop", use: { ...devices["Desktop Chrome"], viewport: { width: 1440, height: 1000 } } },
    { name: "mobile", use: { ...devices["Pixel 7"] }, grep: /@mobile/ },
  ],
});
