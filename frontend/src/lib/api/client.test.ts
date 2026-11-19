import { afterEach, describe, expect, it, vi } from "vitest";
import { api, ApiError, isPlayerId } from "./client";
import { retryDelay, retryPolicy, WAKE_RETRIES } from "./queries";
import { fmtProb, humanize, sentence } from "@/lib/format";

afterEach(() => vi.unstubAllGlobals());

const respond = (status: number, body: unknown) =>
  vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify(body), { status })));

describe("API client", () => {
  it("builds typed requests with encoded parameters", async () => {
    const f = vi.fn(async () => new Response(JSON.stringify({ items: [] }), { status: 200 }));
    vi.stubGlobal("fetch", f);
    await api.searchProspects({ q: "  josh allen ", draftYear: 2018, limit: 5 });
    await api.forecast("qb_226", "post_draft");
    const urls = f.mock.calls.map((c) => String((c as unknown[])[0]));
    expect(urls[0]).toContain("/api/prospects?q=josh+allen&draft_year=2018&limit=5&offset=0");
    expect(urls[1]).toContain("/api/prospects/qb_226/forecast?mode=post_draft");
  });

  it("maps backend error bodies to typed errors", async () => {
    respond(409, { error: { code: "AMBIGUOUS_PLAYER", message: "2 prospects share that name", details: { candidates: [1, 2] } } });
    const e = await api.resolveProspect("Sam Smith").catch((x) => x);
    expect(e).toBeInstanceOf(ApiError);
    expect(e.code).toBe("AMBIGUOUS_PLAYER");
    expect(e.status).toBe(409);
    expect(e.details.candidates).toHaveLength(2);
    expect(e.retryable).toBe(false);
  });

  it("reports an unreachable backend instead of inventing data", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => Promise.reject(new TypeError("Failed to fetch"))));
    const e = await api.forecast("qb_1", "pre_draft").catch((x) => x);
    expect(e.code).toBe("BACKEND_UNREACHABLE");
    expect(e.retryable).toBe(true);
  });

  it("flags invalid prediction modes from the backend", async () => {
    respond(422, { error: { code: "INVALID_PREDICTION_MODE", message: "Invalid request parameters.", details: {} } });
    const e = await api.forecast("qb_1", "pre_draft").catch((x) => x);
    expect(e.code).toBe("INVALID_PREDICTION_MODE");
  });

  it("retries only transient failures, at most twice", () => {
    const transient = new ApiError("INTERNAL_ERROR", "x", 500);
    const client = new ApiError("PROSPECT_NOT_FOUND", "x", 404);
    expect(retryPolicy(0, transient)).toBe(true);
    expect(retryPolicy(2, transient)).toBe(false);
    expect(retryPolicy(0, client)).toBe(false);
  });

  it("keeps retrying while a sleeping free-tier backend wakes, then gives up", () => {
    const asleep = new ApiError("BACKEND_UNREACHABLE", "x", 0);
    const gateway = new ApiError("UNEXPECTED_RESPONSE", "x", 502);
    expect(retryPolicy(4, asleep)).toBe(true);
    expect(retryPolicy(WAKE_RETRIES, asleep)).toBe(false);
    expect(retryPolicy(4, gateway)).toBe(true);
    expect(retryPolicy(0, new ApiError("MODEL_UNAVAILABLE", "x", 503))).toBe(false);   // a missing bundle will not fix itself
    expect([0, 1, 2, 3, 4].map(retryDelay)).toEqual([2000, 4000, 8000, 15000, 15000]);
  });

  it("times out a hung request as unreachable (retryable) instead of waiting forever", async () => {
    const hang: typeof fetch = (_u, init) =>
      new Promise((_, reject) => init?.signal?.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError"))));
    const e = await api.health({ fetcher: hang, timeoutMs: 20 }).catch((x) => x);
    expect(e).toBeInstanceOf(ApiError);
    expect(e.code).toBe("BACKEND_UNREACHABLE");
    expect(e.message).toMatch(/too long/);
    expect(e.retryable).toBe(true);
  });

  it("propagates a caller's abort untouched (React Query cancellation)", async () => {
    const hang: typeof fetch = (_u, init) =>
      new Promise((_, reject) => init?.signal?.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError"))));
    const ctrl = new AbortController();
    const p = api.health({ fetcher: hang, signal: ctrl.signal }).catch((x) => x);
    ctrl.abort();
    const e = await p;
    expect(e).not.toBeInstanceOf(ApiError);
    expect(e.name).toBe("AbortError");
  });

  it("validates player ids before routing", () => {
    expect(isPlayerId("qb_319")).toBe(true);
    expect(isPlayerId("cfb_4606194_2024")).toBe(true);
    expect(isPlayerId("../etc")).toBe(false);
    expect(isPlayerId("QB-1")).toBe(false);
  });
});

describe("formatting", () => {
  it("shows extreme probabilities as bounds, never 0% or 100%", () => {
    expect(fmtProb(0.9999)).toBe(">99%");
    expect(fmtProb(0.0001)).toBe("<1%");
    expect(fmtProb(0.523)).toBe("52%");
    expect(fmtProb(null)).toBe("n/a");
  });
  it("turns backend fragments and codes into sentences", () => {
    expect(sentence("seasonal shape from the simulation")).toBe("Seasonal shape from the simulation.");
    expect(humanize("outside the universe (no_evidence_of_four_college_years)")).toBe("Outside the universe (no evidence of four college years).");
  });
});
