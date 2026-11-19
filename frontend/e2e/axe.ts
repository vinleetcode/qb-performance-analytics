/** Minimal axe-core runner for Playwright (injects the locally installed axe-core; no network). */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import type { Page } from "@playwright/test";

const AXE_SOURCE = readFileSync(join(process.cwd(), "node_modules", "axe-core", "axe.min.js"), "utf-8");

export interface AxeViolation {
  id: string;
  impact: string | null;
  nodes: { target: string[] }[];
}

export default class AxeBuilder {
  constructor(private page: Page) {}
  async analyze(): Promise<AxeViolation[]> {
    await this.page.addScriptTag({ content: AXE_SOURCE });
    return this.page.evaluate(async () => {
      // @ts-expect-error injected global
      const r = await window.axe.run(document, { resultTypes: ["violations"] });
      return r.violations.map((v: AxeViolation) => ({ id: v.id, impact: v.impact, nodes: v.nodes.map((n) => ({ target: n.target })) }));
    });
  }
}
