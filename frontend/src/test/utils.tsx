import { render } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { vi } from "vitest";
import axe from "axe-core";
import { TooltipProvider } from "@/components/ui/tooltip";

export function renderWithClient(ui: React.ReactElement) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  return render(
    <QueryClientProvider client={client}>
      <TooltipProvider>{ui}</TooltipProvider>
    </QueryClientProvider>,
  );
}

/** Stub fetch: `routes` maps a path prefix to [status, body]. Unmatched paths fail loudly. */
export function mockFetch(routes: Record<string, [number, unknown]>) {
  const calls: string[] = [];
  const fn = vi.fn(async (input: RequestInfo | URL) => {
    const url = new URL(String(input));
    calls.push(url.pathname + url.search);
    const hit = Object.keys(routes)
      .sort((a, b) => b.length - a.length)
      .find((k) => url.pathname.startsWith(k));
    if (!hit) throw new Error(`unexpected request ${url}`);
    const [status, body] = routes[hit];
    return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
  });
  vi.stubGlobal("fetch", fn);
  return { fn, calls };
}

/** axe-core in jsdom: structural rules only (jsdom cannot compute colour contrast). */
export async function axeViolations(container: HTMLElement) {
  const res = await axe.run(container, { rules: { "color-contrast": { enabled: false }, region: { enabled: false } } });
  return res.violations.map((v) => `${v.id}: ${v.nodes.length} node(s)`);
}
