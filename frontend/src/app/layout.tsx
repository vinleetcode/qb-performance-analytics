import type { Metadata, Viewport } from "next";
import { Barlow_Condensed, Public_Sans } from "next/font/google";
import { Providers } from "@/components/providers";
import { SiteFooter, SiteHeader } from "@/components/site-chrome";
import "./globals.css";

const display = Barlow_Condensed({ variable: "--font-display", subsets: ["latin"], weight: ["500", "600", "700"] });
const body = Public_Sans({ variable: "--font-body", subsets: ["latin"] });

const DESCRIPTION =
  "Draft projections, NFL starting chances and five-season production forecasts for college quarterbacks, with prediction intervals and honest backtests.";

// Absolute base for Open Graph links: the explicit site URL, else Vercel's production domain (a system variable set at
// build time, so no domain is hardcoded), else localhost for development.
const SITE_URL =
  process.env.NEXT_PUBLIC_SITE_URL ||
  (process.env.VERCEL_PROJECT_PRODUCTION_URL ? `https://${process.env.VERCEL_PROJECT_PRODUCTION_URL}` : "http://localhost:3000");

export const metadata: Metadata = {
  metadataBase: new URL(SITE_URL),
  title: { default: "QB Forecast Lab: from college production to NFL projection", template: "%s | QB Forecast Lab" },
  description: DESCRIPTION,
  applicationName: "QB Forecast Lab",
  openGraph: { type: "website", siteName: "QB Forecast Lab", title: "QB Forecast Lab", description: DESCRIPTION },
  twitter: { card: "summary_large_image", title: "QB Forecast Lab", description: DESCRIPTION },
};

export const viewport: Viewport = { themeColor: "#0f1a2b", colorScheme: "dark" };

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${display.variable} ${body.variable} dark h-full antialiased`}>
      <body className="flex min-h-full flex-col">
        <Providers>
          <SiteHeader />
          <main id="main" className="flex-1">
            {children}
          </main>
          <SiteFooter />
        </Providers>
      </body>
    </html>
  );
}
