import "@testing-library/jest-dom/vitest";
import { afterEach, vi } from "vitest";
import { cleanup } from "@testing-library/react";

afterEach(() => cleanup());

// Recharts' ResponsiveContainer needs ResizeObserver, which jsdom does not provide.
class RO {
  observe() {}
  unobserve() {}
  disconnect() {}
}
vi.stubGlobal("ResizeObserver", RO);

// next/navigation outside the App Router runtime
export const routerPush = vi.fn();
export const routerReplace = vi.fn();
vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: routerPush, replace: routerReplace, prefetch: vi.fn(), back: vi.fn() }),
  usePathname: () => "/test",
  useSearchParams: () => new URLSearchParams(),
  notFound: vi.fn(),
}));
