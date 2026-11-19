import type { NextConfig } from "next";

// On Vercel (preview and production) the API URL must be set explicitly and use HTTPS: a missing value would silently
// compile http://localhost:8000 into the browser bundle and the CSP.
if (process.env.VERCEL) {
  const v = process.env.NEXT_PUBLIC_API_BASE_URL;
  if (!v || !v.startsWith("https://")) {
    throw new Error("NEXT_PUBLIC_API_BASE_URL must be set to the backend's https:// URL in the Vercel project settings.");
  }
}

const api = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
const apiOrigin = (() => {
  try {
    return new URL(api).origin;
  } catch {
    return "";
  }
})();

// Security headers for every route. The CSP allows only this site and the forecasting API: no third-party scripts,
// fonts or images are loaded at runtime (fonts are self-hosted by next/font, photos are local).
const csp = [
  "default-src 'self'",
  "script-src 'self' 'unsafe-inline'" + (process.env.NODE_ENV === "development" ? " 'unsafe-eval'" : ""),
  "style-src 'self' 'unsafe-inline'",
  "img-src 'self' data: blob:",
  "font-src 'self'",
  `connect-src 'self' ${apiOrigin}`.trim(),
  "frame-ancestors 'none'",
  "base-uri 'self'",
  "form-action 'self'",
  "object-src 'none'",
].join("; ");

const securityHeaders = [
  { key: "Content-Security-Policy", value: csp },
  { key: "X-Content-Type-Options", value: "nosniff" },
  { key: "X-Frame-Options", value: "DENY" },
  { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
  { key: "Permissions-Policy", value: "camera=(), microphone=(), geolocation=(), interest-cohort=()" },
  { key: "Cross-Origin-Opener-Policy", value: "same-origin" },
];

const nextConfig: NextConfig = {
  cacheComponents: true,
  partialPrefetching: true,
  poweredByHeader: false,
  images: { formats: ["image/avif", "image/webp"], minimumCacheTTL: 60 * 60 * 24 * 30 },
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
  async headers() {
    return [{ source: "/:path*", headers: securityHeaders }];
  },
};

export default nextConfig;
