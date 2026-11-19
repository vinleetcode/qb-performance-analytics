import { cn } from "@/lib/utils";

export const BRAND = "QB Forecast Lab";

/**
 * The QB Forecast Lab mark: a quarterback's throwing arc (release point and solid flight), the projected remainder as
 * two dots, and the landing zone drawn as a forecast range, an 80% band with its 50% core, exactly as the field ruler
 * draws every range on the site. Geometry is on a 58 x 40 grid; the standalone SVG files in public/brand use the same
 * paths (see README, "Logo").
 *
 * `tone="light"` is for light backgrounds (print, light previews); the site itself is dark-only.
 */
export function LogoMark({ height = 24, tone = "dark", className }: { height?: number; tone?: "dark" | "light"; className?: string }) {
  const ink = tone === "dark" ? "var(--chalk)" : "#0F1A2B";
  const acc = tone === "dark" ? "var(--projection)" : "#2F6DB5";
  const band = tone === "dark" ? "var(--projection-80)" : "#A9C3E3";
  return (
    <svg width={(height * 58) / 40} height={height} viewBox="0 0 58 40" aria-hidden className={cn("shrink-0", className)}>
      <path d="M5 29Q19.4 9.8 33.8 13.64" fill="none" stroke={ink} strokeWidth="3.8" strokeLinecap="round" />
      <circle cx="41" cy="17" r="2.05" fill={acc} />
      <circle cx="47.24" cy="22.24" r="2.05" fill={acc} />
      <rect x="42" y="31" width="15" height="5.2" rx="1.4" fill={band} />
      <rect x="46" y="31" width="7.6" height="5.2" rx="1.4" fill={acc} />
      <circle cx="5" cy="29" r="3.4" fill={ink} />
    </svg>
  );
}

/**
 * Horizontal lockup. The mark is sized from the type: its range band sits on the baseline and the arc tops out at cap
 * height, so the two read as one unit at any size.
 */
export function Logo({ size = 22, className }: { size?: number; className?: string }) {
  return (
    <span className={cn("inline-flex items-end gap-[0.34em] font-display font-semibold leading-none text-chalk", className)} style={{ fontSize: size }}>
      {/* with line-height 1 the baseline sits 0.1em above the line box; at 1.067em the band bottom lands on it */}
      <LogoMark height={size * 1.067} />
      <span className="tracking-[0.005em]">{BRAND}</span>
    </span>
  );
}
