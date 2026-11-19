/** Public runtime configuration. Only non-secret values belong here: anything prefixed NEXT_PUBLIC_ is sent to browsers. */
export const API_BASE_URL = (process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000").replace(/\/$/, "");
export const GITHUB_URL = process.env.NEXT_PUBLIC_GITHUB_URL ?? "https://github.com/vinleetcode/qb-performance-analytics";
