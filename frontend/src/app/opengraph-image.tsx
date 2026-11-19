import { ImageResponse } from "next/og";

// Original social preview: the logo mark and the product line. No photos or data, so nothing to license.
export const alt = "QB Forecast Lab: from college production to NFL projection";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

export default function OpengraphImage() {
  return new ImageResponse(
    (
      <div style={{ width: "100%", height: "100%", display: "flex", flexDirection: "column", justifyContent: "space-between", background: "#0f1a2b", padding: 72, color: "#e9e5db" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 20, fontSize: 34, letterSpacing: 1 }}>
          {/* the logo mark, same geometry as src/components/brand.tsx */}
          <svg width="81" height="56" viewBox="0 0 58 40">
            <path d="M5 29Q19.4 9.8 33.8 13.64" fill="none" stroke="#e9e5db" strokeWidth="3.8" strokeLinecap="round" />
            <circle cx="41" cy="17" r="2.05" fill="#4c8bd1" />
            <circle cx="47.24" cy="22.24" r="2.05" fill="#4c8bd1" />
            <rect x="42" y="31" width="15" height="5.2" rx="1.4" fill="#2e527e" />
            <rect x="46" y="31" width="7.6" height="5.2" rx="1.4" fill="#4c8bd1" />
            <circle cx="5" cy="29" r="3.4" fill="#e9e5db" />
          </svg>
          QB Forecast Lab
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
          <div style={{ fontSize: 84, lineHeight: 1, fontWeight: 700, maxWidth: 980 }}>From college production to NFL projection.</div>
          <div style={{ fontSize: 30, color: "#a3aec0", maxWidth: 960 }}>
            Draft projections, NFL starting chances and five-season forecasts for quarterbacks, tested against real careers.
          </div>
        </div>
        <div style={{ display: "flex", height: 26, width: "100%", background: "#15233a", borderRadius: 4, position: "relative" }}>
          <div style={{ position: "absolute", left: "18%", width: "56%", height: 26, background: "#233d60", borderRadius: 4 }} />
          <div style={{ position: "absolute", left: "26%", width: "40%", height: 26, background: "#2e527e", borderRadius: 4 }} />
          <div style={{ position: "absolute", left: "34%", width: "22%", height: 26, background: "#3e71ab", borderRadius: 4 }} />
          <div style={{ position: "absolute", left: "44%", top: -8, width: 7, height: 42, background: "#e9e5db", borderRadius: 4 }} />
        </div>
      </div>
    ),
    size,
  );
}
