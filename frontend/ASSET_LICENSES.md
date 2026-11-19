# Asset licenses

Every image, font and icon set the site ships with. Machine-readable image metadata (pulled from the Wikimedia Commons API
on the date obtained) is in [`public/images/licenses.json`](public/images/licenses.json). The site's
[`/credits`](src/app/credits/page.tsx) page shows the required attributions, and every photo also has an on-image
credit.

## Photographs

Selection rules:

* No photo shows an identifiable athlete as the subject, NFL marks or team logos.
* Photos were chosen from Wikimedia Commons because each file has verifiable author and licence metadata.
* Unsplash's search could not be queried by script, and Pexels needs an API key, so no images came from either.
* Share-alike (CC BY-SA) images were avoided.

| File (`public/images/`) | Used on | Title on Wikimedia Commons | Creator | Licence | Attribution required | Obtained |
|---|---|---|---|---|---|---|
| `hero-stadium-night.jpg` | Homepage hero | [Roosevelt Stadium night 2](https://commons.wikimedia.org/wiki/File:Roosevelt_Stadium_night_2.jpg) | JonRidinger (own work, 2008) | [CC BY 3.0](https://creativecommons.org/licenses/by/3.0) | yes | 2026-11-11 |
| `yard-line-50.jpg` | Prospect explorer | [50 Yard line at Robert Kraft Field at Wien Stadium](https://commons.wikimedia.org/wiki/File:50_Yard_line_at_Robert_Kraft_Field_at_Wien_Stadium.jpg) | Kenneth C. Zirkel (own work, 2024) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0) | yes | 2026-11-11 |
| `stadium-lights.jpg` | Scouting report header | [George Finnie Stadium at Night](https://commons.wikimedia.org/wiki/File:George_Finnie_Stadium_at_Night_(6253229054).jpg) | Erik Drost (2010) | [CC BY 2.0](https://creativecommons.org/licenses/by/2.0) | yes | 2026-11-11 |
| `goal-line-1922.jpg` | Backtesting | [Goal-line action at Miami-Akron football game 1922](https://commons.wikimedia.org/wiki/File:Goal-line_action_at_Miami-Akron_football_game_1922_(3191455980).jpg) | Frank R. Snyder (via Miami University Libraries) | Public domain (published 1922) | no | 2026-11-11 |
| `practice-field.jpg` | Methodology | [Stanford football practice field, Dan Elliot field](https://commons.wikimedia.org/wiki/File:Stanford_football_practice_field_-_Dan_Elliot_field_-_panoramio.jpg) | Thomson M (2015) | [CC BY 3.0](https://creativecommons.org/licenses/by/3.0) | yes | 2026-11-11 |
| `line-play-1927.jpg` | Case study | [Chicago's Leyers picks a hole in the Michigan line (1927)](https://commons.wikimedia.org/wiki/File:Chicago%27s_Leyers_picks_a_hole_in_the_Michigan_Line_for_a_five_yard_gain_(1927).jpg) | Chicago Tribune, 6 Nov 1927 | Public domain (US work published 1927) | no | 2026-11-11 |

Modifications: the images were resized (2000 px maximum) and recompressed as progressive JPEG. On the site they are cropped
and darkened with overlays. CC BY allows this; the credits note that the images were modified.

Required attribution text, as shown on `/credits` and on each image:

* "Roosevelt Stadium night 2" by JonRidinger, CC BY 3.0, via Wikimedia Commons (cropped, darkened)
* "50 Yard line at Robert Kraft Field at Wien Stadium" by Kenneth C. Zirkel, CC BY 4.0, via Wikimedia Commons (cropped, darkened)
* "George Finnie Stadium at Night" by Erik Drost, CC BY 2.0, via Wikimedia Commons (cropped, darkened)
* "Stanford football practice field" by Thomson M, CC BY 3.0, via Wikimedia Commons (cropped, darkened)
* Public domain: Frank R. Snyder (1922, Miami University Libraries); Chicago Tribune (1927). Credited as a courtesy.

### Rights the licences do not cover

Copyright licences and public-domain status do not grant trademark or publicity rights.

* **People.** No image has an identifiable present-day person as its subject. The 1922 and 1927 photos show players from
  about a century ago, in public-domain news and archive photographs. Spectators in the stadium shots are distant and not
  identifiable.
* **Marks.** Stadium photos may contain small incidental signage. None shows an NFL shield, team logo or school logo as a
  feature, and the site does not imply endorsement by any school, league or team.
* **Players' own photos.** Prospect reports use initials and typography, never player photographs, because no image with
  verified rights for any specific player was found.

## Images rejected during selection

| Candidate | Licence | Reason |
|---|---|---|
| American football ball (Petey21) | CC0 | NFL shield and Wilson marks prominent |
| Football on ground (Dsw4) | public domain | Nike mark prominent |
| Fifth Third Stadium at Night | CC BY 4.0 | Identifiable spectators in the foreground |
| Aerial view of Neyland Stadium | CC BY 4.0 | School logos and stadium branding prominent |
| A night game at Princeton Stadium | CC BY 4.0 | School name painted in the end zone |

## Fonts

| Font | Licence | Delivery |
|---|---|---|
| Barlow Condensed | SIL Open Font License 1.1 | `next/font/google`: downloaded at build time and self-hosted from the site |
| Public Sans | SIL Open Font License 1.1 | same |

## Icons and generated graphics

* Lucide icons: ISC licence.
* The logo (mark, horizontal lockups, favicon and app icons in `public/brand/`, `src/app/icon.svg`,
  `src/app/apple-icon.png`), the Open Graph image and all diagrams (pipeline and timeline graphics) are original work made
  for this project, in SVG, CSS or code. No NFL, NCAA, school or team mark was used or imitated.
* The wordmark in the standalone logo SVGs is Barlow Condensed SemiBold converted to outlines
  (`scripts/build-brand.py`). The SIL Open Font License allows this; the files embed shapes, not the font.
* Product screenshots are captures of this application.

## Data shown on the site

Data attributions are on `/credits`: CollegeFootballData.com (attribution recommended) and nflverse (CC BY 4.0). See
[`../nfl-qb-analytics/reports/public_data_licensing_review.md`](../nfl-qb-analytics/reports/public_data_licensing_review.md).
