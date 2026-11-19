/**
 * Licensed photography (see ASSET_LICENSES.md). Every entry carries the credit that CC BY requires; the PhotoBanner
 * component always renders it. Only images listed here may be used on the site.
 */
export interface SitePhoto {
  src: string;
  alt: string;
  width: number;
  height: number;
  credit: string;
  license: string;
  source: string;
  position?: string;
}

const commons = (file: string) => `https://commons.wikimedia.org/wiki/File:${file}`;

export const PHOTOS = {
  heroStadium: {
    src: "/images/hero-stadium-night.jpg",
    alt: "A high-school football stadium at night, floodlights glowing above a dark sky and a distant field.",
    width: 2000,
    height: 932,
    credit: "JonRidinger",
    license: "CC BY 3.0",
    source: commons("Roosevelt_Stadium_night_2.jpg"),
    position: "50% 40%",
  },
  yardLine: {
    src: "/images/yard-line-50.jpg",
    alt: "The painted 50-yard-line numbers on a grass football field, stands and floodlight towers behind.",
    width: 2000,
    height: 1298,
    credit: "Kenneth C. Zirkel",
    license: "CC BY 4.0",
    source: commons("50_Yard_line_at_Robert_Kraft_Field_at_Wien_Stadium.jpg"),
    position: "50% 65%",
  },
  stadiumLights: {
    src: "/images/stadium-lights.jpg",
    alt: "A floodlight tower and press box above a football field during a night game.",
    width: 1747,
    height: 1251,
    credit: "Erik Drost",
    license: "CC BY 2.0",
    source: commons("George_Finnie_Stadium_at_Night_(6253229054).jpg"),
    position: "50% 80%",
  },
  goalLine1922: {
    src: "/images/goal-line-1922.jpg",
    alt: "A black-and-white 1922 photograph of a college football goal-line play, players piled at the line.",
    width: 2000,
    height: 1436,
    credit: "Frank R. Snyder, Miami University Libraries (1922)",
    license: "Public domain",
    source: commons("Goal-line_action_at_Miami-Akron_football_game_1922_(3191455980).jpg"),
    position: "50% 45%",
  },
  practiceField: {
    src: "/images/practice-field.jpg",
    alt: "An empty football practice field in daylight, floodlight poles along the far sideline.",
    width: 2000,
    height: 1333,
    credit: "Thomson M",
    license: "CC BY 3.0",
    source: commons("Stanford_football_practice_field_-_Dan_Elliot_field_-_panoramio.jpg"),
    position: "50% 60%",
  },
  linePlay1927: {
    src: "/images/line-play-1927.jpg",
    alt: "A 1927 newspaper photograph of a ball carrier running through the line in a college football game.",
    width: 2000,
    height: 650,
    credit: "Chicago Tribune (1927)",
    license: "Public domain",
    source: commons("Chicago%27s_Leyers_picks_a_hole_in_the_Michigan_Line_for_a_five_yard_gain_(1927).jpg"),
    position: "50% 50%",
  },
} satisfies Record<string, SitePhoto>;

export const creditLine = (p: SitePhoto) => `Photo: ${p.credit}, ${p.license}, via Wikimedia Commons`;
