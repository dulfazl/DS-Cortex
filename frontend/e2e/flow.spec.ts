import path from "node:path";
import { expect, test } from "@playwright/test";

const CSV = path.join(__dirname, "customers.csv");

test("upload, review-mode approvals, results tabs and downloads", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId("file-input").setInputFiles(CSV);
  await page.getByLabel("What do you want to know?").fill("Predict whether a customer will churn and explain the drivers");
  await page.getByText("Review", { exact: true }).click();
  await page.getByRole("button", { name: "Upload and profile" }).click();

  await expect(page.getByRole("heading", { name: "Quick profile" })).toBeVisible();
  await expect(page.getByText("possible target")).toBeVisible();
  await page.getByRole("button", { name: "Start" }).click();

  // approve each stage bundle until the run completes
  const approve = page.getByRole("button", { name: "Approve", exact: true, disabled: false });
  const bundle = page.getByRole("heading", { name: / bundle$/ });
  const done = page.getByRole("tab", { name: "Overview" });
  for (let i = 0; i < 6; i++) {
    await expect(approve.or(done)).toBeVisible();
    if (await done.isVisible()) break;
    const [title] = await bundle.allTextContents();
    await approve.click();
    // wait until this bundle is gone (the next one, or the results). allTextContents() never waits for an
    // element, unlike textContent(), which would hang if the heading vanished between two checks.
    await expect
      .poll(async () => {
        if (await done.isVisible()) return true;
        const titles = await bundle.allTextContents();
        return titles.length === 0 || titles[0] !== title;
      })
      .toBe(true);
  }
  await expect(done).toBeVisible();
  await expect(page.getByText(/Held-out f1_macro/)).toBeVisible();

  for (const tab of ["Plots", "Model", "Decision Ledger", "Debate Transcript", "Code"]) {
    await page.getByRole("tab", { name: tab }).click();
  }
  await expect(page.getByText("pipeline.py: reproduces the held-out score")).toBeVisible();
  const [download] = await Promise.all([page.waitForEvent("download"), page.getByRole("link", { name: "Everything (zip)" }).click()]);
  expect(download.suggestedFilename()).toMatch(/\.zip$/);

  // evidence drawer opens from a confidence badge
  await page.getByRole("tab", { name: "Decision Ledger" }).click();
  await page.getByTitle("Show the evidence behind this confidence").first().click();
  await expect(page.getByRole("dialog", { name: "Evidence" })).toBeVisible();

  // no horizontal page scroll at any width
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  expect(overflow).toBeLessThanOrEqual(1);
});
