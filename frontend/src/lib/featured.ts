/**
 * Curated starting points. Ids are the stable `qb_draft_prospect_id`s of the backend dataset; every value shown
 * for them is fetched from the API. Chosen for recognisability and for a spread of outcomes (stars and misses),
 * not for how accurate their forecasts happened to be.
 */
export const BACKTEST_EXAMPLES = [
  { id: "qb_226", name: "Josh Allen", year: 2018 },
  { id: "qb_215", name: "Patrick Mahomes", year: 2017 },
  { id: "qb_248", name: "Joe Burrow", year: 2020 },
  { id: "qb_227", name: "Josh Rosen", year: 2018 },
  { id: "qb_179", name: "Johnny Manziel", year: 2014 },
  { id: "qb_206", name: "Dak Prescott", year: 2016 },
  { id: "qb_262", name: "Zach Wilson", year: 2021 },
  { id: "qb_281", name: "C.J. Stroud", year: 2023 },
  { id: "qb_107", name: "Matt Ryan", year: 2008 },
] as const;
