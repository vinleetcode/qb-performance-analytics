import { describe, expect, it } from "vitest";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { StatForecast } from "@/lib/api/types";
import { HeadlineStats, IntervalExplorer, ProjectionTable, ReadingGuide } from "./production-forecast";
import { InfoTip } from "@/components/info-tip";
import { intervalsNested, readCoverage } from "@/lib/forecast";
import { axeViolations } from "@/test/utils";
import * as F from "@/test/fixtures";

const totals = () => F.forecast().career.five_year_totals;
const byStat = (xs: StatForecast[]) => new Map(xs.map((s) => [s.stat, s]));
const withCoverage = (xs: StatForecast[], stat: string, c80: number) =>
  xs.map((s) => (s.stat === stat ? { ...s, measured_coverage: { c50: 0.45, c80, c90: 0.8 } } : s));

describe("headline statistics", () => {
  it("leads with the five headline totals, their 80% ranges and the five-season horizon", () => {
    render(<HeadlineStats stats={byStat(totals())} />);
    const list = screen.getByRole("list", { name: "Headline forecast, NFL seasons 1 to 5" });
    const items = within(list).getAllByRole("listitem");
    expect(items.map((li) => li.querySelector("span")?.textContent)).toEqual([
      "Passing yards", "Passing TDs", "Interceptions", "Rushing yards", "Games started",
    ]);
    expect(within(items[0]).getByText("7,111")).toBeInTheDocument();
    expect(within(items[0]).getByText("2,133 to 13,511")).toBeInTheDocument();
    expect(items[0]).toHaveTextContent("expected yards");
    expect(screen.getByText(/first five NFL seasons/)).toBeInTheDocument();
  });

  it("shows a missing statistic as unavailable, never as a number", () => {
    render(<HeadlineStats stats={byStat(totals().filter((s) => s.stat !== "rush_yards"))} />);
    const items = screen.getAllByRole("listitem");
    expect(within(items[3]).getByText("Not available for this player")).toBeInTheDocument();
    expect(within(items[3]).queryByText(/\d/)).not.toBeInTheDocument();
  });

  it("flags weak backtest coverage on the card", () => {
    render(<HeadlineStats stats={byStat(withCoverage(totals(), "rush_yards", 0.682))} />);
    const items = screen.getAllByRole("listitem");
    expect(within(items[3]).getByText("Runs narrow")).toBeInTheDocument();
    expect(within(items[3]).getByText(/68% coverage/)).toBeInTheDocument();
    expect(within(items[0]).queryByText("Runs narrow")).not.toBeInTheDocument();
  });
});

describe("interval explorer", () => {
  it("draws the selected statistic's 50% and 80% ranges with units, and switches statistic", async () => {
    render(<IntervalExplorer stats={byStat(totals())} />);
    const chart = () => screen.getByRole("img", { name: /^Range chart for/ });
    expect(chart().getAttribute("aria-label")).toBe(
      "Range chart for passing yards over NFL seasons 1 to 5: expected 7,111 yards; 50% range 4,267 to 9,955; 80% range 2,133 to 13,511. Scale 0 to 20,000 yards.",
    );
    expect(screen.getByText("expected yards, NFL seasons 1 to 5")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("radio", { name: "Games started" }));
    expect(chart().getAttribute("aria-label")).toMatch(/^Range chart for games started over NFL seasons 1 to 5: expected 30 starts; 50% range 18 to 42; 80% range 9 to 57/);
    // arrow keys move within the radio group
    screen.getByRole("radio", { name: "Games started" }).focus();
    await userEvent.keyboard("{ArrowLeft}");
    expect(screen.getByRole("radio", { name: "Rushing yards" })).toBeChecked();
  });

  it("warns when a range ran narrow in backtests and when ranges are not nested", async () => {
    const xs = withCoverage(totals(), "pass_yards", 0.7).map((s) =>
      s.stat === "pass_tds" ? { ...s, intervals: [{ level: 0.5, lower: 10, upper: 90 }, { level: 0.8, lower: 20, upper: 80 }] } : s,
    );
    render(<IntervalExplorer stats={byStat(xs)} />);
    expect(screen.getByText(/Range ran narrow in backtests: 70% of real totals fell inside/)).toBeInTheDocument();
    await userEvent.click(screen.getByRole("radio", { name: "Passing TDs" }));
    expect(screen.getByRole("status")).toHaveTextContent("not inside its 80% range");
  });
});

describe("full projection table", () => {
  it("keeps all ten projections, grouped, with explained column headers", async () => {
    render(<ProjectionTable stats={totals()} />);
    const table = screen.getByRole("table", { name: /All ten projections/ });
    expect(within(table).getAllByRole("row")).toHaveLength(11);
    for (const h of ["Expected", "50% range", "80% range", "80% coverage in backtests"])
      expect(within(table).getByRole("button", { name: `About ${h}` })).toBeInTheDocument();
    expect(within(table).getAllByRole("rowheader").map((r) => r.textContent)).toEqual([
      "PassingPassing yards", "Passing TDs", "Interceptions", "Pass attempts", "Completions",
      "RushingRushing yards", "Rushing TDs", "Rush attempts", "Playing timeGames started", "Games played",
    ]);
    expect(await axeViolations(table)).toEqual([]);
  });

  it("marks weak coverage in the table and in the reading guide", () => {
    const xs = withCoverage(withCoverage(totals(), "rush_tds", 0.682), "rush_yards", 0.729);
    render(<><ProjectionTable stats={xs} /><ReadingGuide stats={xs} /></>);
    const table = screen.getByRole("table");
    expect(within(table).getAllByText("Runs narrow")).toHaveLength(2);
    expect(screen.getByText(/the ranges for rushing yards and rushing TDs caught fewer than 75%/)).toBeInTheDocument();
  });
});

describe("toggletip", () => {
  it("opens with a click or tap, with the keyboard, and closes with Escape", async () => {
    render(<InfoTip label="80% range">A wider range.</InfoTip>);
    const btn = screen.getByRole("button", { name: "About 80% range" });
    await userEvent.click(btn);
    expect(await screen.findByText("A wider range.")).toBeInTheDocument();
    await userEvent.keyboard("{Escape}");
    await expect.poll(() => screen.queryByText("A wider range.")).toBeNull();
    btn.focus();
    await userEvent.keyboard("{Enter}");
    expect(await screen.findByText("A wider range.")).toBeInTheDocument();
  });
});

describe("forecast helpers", () => {
  it("reads coverage against documented thresholds", () => {
    expect(readCoverage(0.682)).toBe("narrow");
    expect(readCoverage(0.75)).toBe("on_target");
    expect(readCoverage(0.748)).toBe("on_target"); // shown as 75%, so not flagged as under 75%
    expect(readCoverage(0.744)).toBe("narrow");
    expect(readCoverage(0.85)).toBe("on_target");
    expect(readCoverage(0.9)).toBe("wide");
    expect(readCoverage(null)).toBeNull();
  });
  it("checks that the 50% range lies inside the 80% range", () => {
    for (const s of totals()) expect(intervalsNested(s)).toBe(true);
    expect(intervalsNested({ ...totals()[0], intervals: [{ level: 0.5, lower: 1, upper: 9 }, { level: 0.8, lower: 2, upper: 8 }] })).toBe(false);
  });
});
