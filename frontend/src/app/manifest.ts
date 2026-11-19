import type { MetadataRoute } from "next";

export default function manifest(): MetadataRoute.Manifest {
  return {
    name: "QB Forecast Lab",
    short_name: "QB Forecast Lab",
    description: "Quarterback draft and NFL career forecasts with honest ranges and historical backtests.",
    start_url: "/",
    display: "standalone",
    background_color: "#0f1a2b",
    theme_color: "#0f1a2b",
    icons: [
      { src: "/brand/icon-192.png", sizes: "192x192", type: "image/png" },
      { src: "/brand/icon-512.png", sizes: "512x512", type: "image/png" },
      { src: "/brand/app-icon.svg", sizes: "any", type: "image/svg+xml", purpose: "maskable" },
    ],
  };
}
