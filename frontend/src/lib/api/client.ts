import { API_BASE_URL } from "@/lib/config";
import type {
  BacktestResponse,
  ComparisonsResponse,
  ForecastResponse,
  HealthResponse,
  Mode,
  ModelsResponse,
  PerformanceResponse,
  ProspectDetail,
  ProspectSearchResponse,
  ProspectSummary,
} from "./types";

/** Stable backend error codes (reports/api_endpoint_reference.md) plus two client-side ones. */
export type ApiErrorCode =
  | "PROSPECT_NOT_FOUND"
  | "AMBIGUOUS_PLAYER"
  | "MODEL_UNAVAILABLE"
  | "PREDICTION_UNSUPPORTED"
  | "INSUFFICIENT_DATA"
  | "INVALID_PREDICTION_MODE"
  | "INVALID_PARAMETER"
  | "RATE_LIMITED"
  | "NOT_FOUND"
  | "INTERNAL_ERROR"
  | "BACKEND_UNREACHABLE"
  | "UNEXPECTED_RESPONSE";

export class ApiError extends Error {
  constructor(
    public code: ApiErrorCode,
    message: string,
    public status: number,
    public details: Record<string, unknown> = {},
  ) {
    super(message);
    this.name = "ApiError";
  }
  /** Worth retrying: the service may be starting or briefly overloaded. Client errors are never retried. */
  get retryable(): boolean {
    return this.code === "BACKEND_UNREACHABLE" || this.status >= 500 || this.code === "RATE_LIMITED";
  }
}

export type Fetcher = typeof fetch;
type Opts = { fetcher?: Fetcher; signal?: AbortSignal; baseUrl?: string; timeoutMs?: number };

/**
 * Per-attempt time limit. The free-tier backend can take about a minute to wake from sleep, so one attempt may wait
 * that long. The query layer then retries with backoff (see queries.ts). It never waits forever.
 */
export const REQUEST_TIMEOUT_MS = 75_000;

async function request<T>(path: string, params: Record<string, string | number | boolean | undefined> = {}, opts: Opts = {}): Promise<T> {
  const base = (opts.baseUrl ?? API_BASE_URL).replace(/\/$/, "");
  const qs = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== "") qs.set(k, String(v));
  const target = `${base}${path}${qs.size ? `?${qs}` : ""}`;
  const ctrl = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    ctrl.abort();
  }, opts.timeoutMs ?? REQUEST_TIMEOUT_MS);
  const forward = () => ctrl.abort();
  opts.signal?.addEventListener("abort", forward, { once: true });
  try {
    let res: Response;
    try {
      res = await (opts.fetcher ?? fetch)(target, { headers: { Accept: "application/json" }, signal: ctrl.signal });
    } catch (e) {
      if (opts.signal?.aborted) throw e;
      throw timedOut
        ? new ApiError("BACKEND_UNREACHABLE", "The forecasting service took too long to respond. It may still be waking up.", 0)
        : new ApiError("BACKEND_UNREACHABLE", "The forecasting service could not be reached.", 0);
    }
    let body: unknown;
    try {
      body = await res.json();
    } catch {
      if (opts.signal?.aborted) throw new DOMException("Aborted", "AbortError");
      throw new ApiError("UNEXPECTED_RESPONSE", `The service returned an unreadable response (HTTP ${res.status}).`, res.status);
    }
    return check<T>(res, body);
  } finally {
    clearTimeout(timer);
    opts.signal?.removeEventListener("abort", forward);
  }
}

function check<T>(res: Response, body: unknown): T {
  if (!res.ok) {
    const err = (body as { error?: { code?: ApiErrorCode; message?: string; details?: Record<string, unknown> } })?.error;
    throw new ApiError(err?.code ?? "UNEXPECTED_RESPONSE", err?.message ?? `Request failed (HTTP ${res.status}).`, res.status, err?.details ?? {});
  }
  return body as T;
}

const enc = encodeURIComponent;

export const api = {
  health: (o?: Opts) => request<HealthResponse>("/api/health", {}, o),
  searchProspects: (p: { q?: string; draftYear?: number; draftedOnly?: boolean; limit?: number; offset?: number }, o?: Opts) =>
    request<ProspectSearchResponse>(
      "/api/prospects",
      {
        q: p.q?.trim() || undefined,
        draft_year: p.draftYear,
        drafted_only: p.draftedOnly || undefined,
        limit: p.limit ?? 20,
        offset: p.offset ?? 0,
      },
      o,
    ),
  resolveProspect: (name: string, draftYear?: number, o?: Opts) =>
    request<ProspectSummary>("/api/prospects/resolve", { name, draft_year: draftYear }, o),
  prospect: (id: string, o?: Opts) => request<ProspectDetail>(`/api/prospects/${enc(id)}`, {}, o),
  forecast: (id: string, mode: Mode, assumedPick?: number, o?: Opts) =>
    request<ForecastResponse>(`/api/prospects/${enc(id)}/forecast`, { mode, assumed_pick: assumedPick }, o),
  comparisons: (id: string, o?: Opts) => request<ComparisonsResponse>(`/api/prospects/${enc(id)}/comparisons`, {}, o),
  backtest: (id: string, o?: Opts) => request<BacktestResponse>(`/api/backtests/${enc(id)}`, {}, o),
  models: (o?: Opts) => request<ModelsResponse>("/api/models", {}, o),
  performance: (o?: Opts) => request<PerformanceResponse>("/api/models/performance", {}, o),
};

/** Player ids are validated before they are used in a URL. */
export const PLAYER_ID_RE = /^(qb_\d{1,6}|cfb_\d{1,12}_\d{4})$/;
export const isPlayerId = (id: string) => PLAYER_ID_RE.test(id);
