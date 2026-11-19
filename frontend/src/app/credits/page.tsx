import type { Metadata } from "next";
import { PHOTOS } from "@/lib/images";

export const metadata: Metadata = { title: "Data and photo credits" };

const DATA = [
  {
    name: "CollegeFootballData.com",
    href: "https://collegefootballdata.com",
    text: "College statistics, games, recruiting and team context. Data provided by CollegeFootballData.com, used under its terms of use (display of factual data and independently generated predictions).",
    terms: "https://collegefootballdata.com/terms",
  },
  {
    name: "nflverse",
    href: "https://github.com/nflverse/nflverse-data",
    text: "Draft results, NFL season statistics, starts and NFL Combine results from the nflverse data releases, licensed CC BY 4.0. Some nflverse datasets originate with third parties, whose own terms may apply.",
    terms: "https://creativecommons.org/licenses/by/4.0/",
  },
  {
    name: "Pro Football Reference (Sports Reference LLC)",
    href: "https://www.pro-football-reference.com",
    text: "nflverse credits its draft-pick and NFL Combine releases to Pro Football Reference. Draft slots and Combine measurements shown here are individual facts from those releases, credited to their original source. This site does not access or scrape Sports Reference sites.",
    terms: "https://www.sports-reference.com/data_use.html",
  },
];

export default function CreditsPage() {
  return (
    <div className="mx-auto max-w-6xl px-4 py-14 sm:px-6">
      <h1 className="text-[clamp(40px,6vw,64px)] text-chalk">Data and photo credits</h1>
      <p className="measure mt-3 text-muted-foreground">
        The data behind the forecasts, the photographs on the site, and the licences they are used under. Model predictions
        are this project&apos;s own work.
      </p>

      <section aria-labelledby="data-h" className="mt-12 border-t border-line pt-8">
        <h2 id="data-h" className="text-[28px] text-chalk">Data</h2>
        <ul className="mt-4 space-y-5">
          {DATA.map((d) => (
            <li key={d.name} className="measure">
              <a href={d.href} target="_blank" rel="noreferrer" className="font-display text-[22px] font-semibold text-chalk underline">
                {d.name}
              </a>
              <p className="mt-1 text-[15px] text-chalk/85">{d.text}</p>
              <a href={d.terms} target="_blank" rel="noreferrer" className="text-[13px] text-projection-text underline">Terms or licence</a>
            </li>
          ))}
        </ul>
      </section>

      <section aria-labelledby="photo-h" className="mt-12 border-t border-line pt-8">
        <h2 id="photo-h" className="text-[28px] text-chalk">Photographs</h2>
        <p className="measure mt-2 text-[14px] text-muted-foreground">
          All via Wikimedia Commons. Resized, cropped and darkened for the site. No photograph shows a current player, and no
          school, team or league endorses this site.
        </p>
        <ul className="mt-5 divide-y divide-line border-y border-line">
          {Object.values(PHOTOS).map((p) => (
            <li key={p.src} className="grid gap-1 py-3 sm:grid-cols-[1fr_auto] sm:gap-6">
              <span className="text-[15px] text-chalk">{p.alt}</span>
              <span className="text-[14px] text-muted-foreground">
                {p.credit}, {p.license}{" "}
                <a href={p.source} target="_blank" rel="noreferrer" className="text-projection-text underline">source</a>
              </span>
            </li>
          ))}
        </ul>
      </section>

      <section aria-labelledby="other-h" className="mt-12 border-t border-line pt-8">
        <h2 id="other-h" className="text-[28px] text-chalk">Fonts and icons</h2>
        <p className="measure mt-2 text-[15px] text-chalk/85">
          Barlow Condensed and Public Sans (SIL Open Font License), self-hosted. Icons from Lucide (ISC licence). The logo, diagrams
          and the preview image are original to this project.
        </p>
      </section>
    </div>
  );
}
