import { defineConfig, devices } from "@playwright/test";

// Expects the API on :8000 and the web app on :3000 (docker compose up, or both dev servers).
export default defineConfig({
  testDir: "./e2e",
  workers: 1,
  // CI runners have 2 cores: one analysis stage can take a few minutes there
  timeout: 900_000,
  expect: { timeout: 300_000 },
  use: { baseURL: process.env.E2E_BASE_URL ?? "http://localhost:3000", trace: "retain-on-failure" },
  projects: [
    { name: "desktop", use: { ...devices["Desktop Chrome"] } },
    { name: "phone", use: { ...devices["Pixel 7"] } },
  ],
});
