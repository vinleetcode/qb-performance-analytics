"use client";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { api, ApiError } from "./client";
import type { Mode } from "./types";

const HOUR = 60 * 60 * 1000;
/**
 * Predictions are deterministic for a given model version, so successful responses stay fresh for an hour.
 * Failures are not cached: React Query stores no data for them and refetches on the next mount. Only transient
 * failures are retried. A sleeping free-tier backend looks like "unreachable" or a 502/503/504 gateway response
 * while it wakes (about a minute), so those get up to five retries with backoff; other transient failures (5xx,
 * rate limited) at most two. Client errors (404, 422, ...) and a missing model bundle are never retried.
 */
export const WAKE_RETRIES = 5;
const isWaking = (e: ApiError) =>
  e.code === "BACKEND_UNREACHABLE" || ((e.status === 502 || e.status === 503 || e.status === 504) && e.code !== "MODEL_UNAVAILABLE");
export const retryPolicy = (count: number, e: unknown) =>
  e instanceof ApiError && e.retryable && e.code !== "MODEL_UNAVAILABLE" && count < (isWaking(e) ? WAKE_RETRIES : 2);
/** 2 s, 4 s, 8 s, then every 15 s: about 45 s of waiting on top of each attempt's own time. */
export const retryDelay = (attempt: number) => Math.min(2000 * 2 ** attempt, 15_000);

export const qk = {
  search: (q: string, year?: number, drafted?: boolean, page = 0, limit = 20) => ["prospects", q, year ?? null, !!drafted, page, limit] as const,
  prospect: (id: string) => ["prospect", id] as const,
  forecast: (id: string, mode: Mode, pick?: number) => ["forecast", id, mode, pick ?? null] as const,
  comparisons: (id: string) => ["comparisons", id] as const,
  backtest: (id: string) => ["backtest", id] as const,
  performance: ["performance"] as const,
};

export function useProspectSearch(
  q: string,
  opts: { draftYear?: number; draftedOnly?: boolean; page?: number; limit?: number; enabled?: boolean } = {},
) {
  const limit = opts.limit ?? 20;
  const page = opts.page ?? 0;
  return useQuery({
    queryKey: qk.search(q, opts.draftYear, opts.draftedOnly, page, limit),
    queryFn: ({ signal }) =>
      api.searchProspects({ q, draftYear: opts.draftYear, draftedOnly: opts.draftedOnly, limit, offset: page * limit }, { signal }),
    staleTime: HOUR,
    retry: retryPolicy,
    placeholderData: keepPreviousData,
    enabled: opts.enabled ?? true,
  });
}

export const useProspect = (id: string) =>
  useQuery({ queryKey: qk.prospect(id), queryFn: () => api.prospect(id), staleTime: HOUR, retry: retryPolicy });

export const useForecast = (id: string, mode: Mode, pick?: number, enabled = true) =>
  useQuery({
    queryKey: qk.forecast(id, mode, pick),
    queryFn: () => api.forecast(id, mode, pick),
    staleTime: HOUR,
    retry: retryPolicy,
    enabled,
  });

export const useComparisons = (id: string) =>
  useQuery({ queryKey: qk.comparisons(id), queryFn: () => api.comparisons(id), staleTime: HOUR, retry: retryPolicy });

export const useBacktest = (id: string | null) =>
  useQuery({
    queryKey: qk.backtest(id ?? ""),
    queryFn: () => api.backtest(id as string),
    staleTime: HOUR,
    retry: retryPolicy,
    enabled: !!id,
  });

export const usePerformance = () =>
  useQuery({ queryKey: qk.performance, queryFn: () => api.performance(), staleTime: HOUR, retry: retryPolicy });
