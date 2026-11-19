import { describe, expect, it, vi, afterEach } from "vitest";
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { FieldRuler } from "@/components/field-ruler";
import { ProspectSearch } from "@/components/prospect-search";
import { ErrorState, Unavailable } from "@/components/status";
import { ApiError } from "@/lib/api/client";
import { DraftSection } from "@/components/report/draft-section";
import { ProductionSection, StartsSection } from "@/components/report/career-sections";
import { ActualDraft, PlayerHeader } from "@/components/report/player-header";
import { ProspectReport } from "@/components/report/prospect-report";
import { UncertaintySection } from "@/components/report/uncertainty-section";
import { BacktestView } from "@/components/backtest/backtest-view";
import { Methodology } from "@/components/methodology/methodology";
import { axeViolations, mockFetch, renderWithClient } from "@/test/utils";
import * as F from "@/test/fixtures";

afterEach(() => vi.unstubAllGlobals());

describe("prospect search", () => {
  it("autocompletes, disambiguates shared names and selects by id with the keyboard", async () => {
    const { calls } = mockFetch({ "/api/prospects": [200, F.searchResponse] });
    const onSelect = vi.fn();
    renderWithClient(<ProspectSearch onSelect={onSelect} />);
    const box = screen.getByRole("combobox");
    await userEvent.type(box, "f");
    await new Promise((r) => setTimeout(r, 300));
    expect(calls).toHaveLength(0); // one character never queries
    await userEvent.type(box, "ixture");
    const list = await screen.findByRole("listbox");
    const opts = await within(list).findAllByRole("option");
    expect(opts).toHaveLength(3);
    expect(within(list).getAllByText("shared name")).toHaveLength(2); // both "Fixture Twin" rows flagged
    expect(within(opts[1]).getByText("North Fixture")).toBeInTheDocument();
    expect(within(opts[2]).getByText("South Fixture")).toBeInTheDocument();
    await userEvent.keyboard("{ArrowDown}{ArrowDown}{Enter}");
    expect(onSelect).toHaveBeenCalledWith(expect.objectContaining({ player_id: "cfb_123456_2020" }));
  });

  it("explains empty results and failures", async () => {
    mockFetch({ "/api/prospects": [200, { ...F.searchResponse, total: 0, items: [] }] });
    renderWithClient(<ProspectSearch />);
    await userEvent.type(screen.getByRole("combobox"), "zzzz");
    expect(await screen.findByText(/No quarterback matches/)).toBeInTheDocument();
  });
});

describe("field ruler", () => {
  it("describes every interval and the actual result in its accessible label", () => {
    render(
      <FieldRuler
        label="Passing yards"
        expected={5000}
        intervals={[
          { level: 0.5, lower: 3000, upper: 7000 },
          { level: 0.8, lower: 2000, upper: 9000 },
        ]}
        actual={8000}
      />,
    );
    const img = screen.getByRole("img");
    const label = img.getAttribute("aria-label")!;
    expect(label).toContain("expected 5,000");
    expect(label).toContain("50% range 3,000 to 7,000");
    expect(label).toContain("80% range 2,000 to 9,000");
    expect(label).toContain("actual 8,000");
  });
});

describe("forecast rendering", () => {
  it("shows an explanation, not a zero, when draft probability is unsupported", () => {
    renderWithClient(<DraftSection f={F.forecast()} />);
    expect(screen.getByText("Probability of being drafted is not available for this player")).toBeInTheDocument();
    expect(screen.getByText(/no evidence of four college years/)).toBeInTheDocument();
    expect(screen.queryByText("0%")).not.toBeInTheDocument();
    expect(screen.getByText("Round 1")).toBeInTheDocument();
    expect(screen.getByText("60%")).toBeInTheDocument();
  });

  it("shows nested starting probabilities with the five-season window", () => {
    renderWithClient(<StartsSection f={F.forecast()} />);
    expect(screen.getByText("71%")).toBeInTheDocument();
    expect(screen.getByText("52%")).toBeInTheDocument();
    expect(screen.getByText("18%")).toBeInTheDocument();
    expect(screen.getByText(/NFL seasons 1-5/)).toBeInTheDocument();
  });

  it("renders every projection with its uncertainty intervals and measured coverage", () => {
    renderWithClient(<ProductionSection f={F.forecast()} />);
    const table = screen.getByRole("table", { name: /All ten projections/ });
    expect(within(table).getAllByRole("row")).toHaveLength(11);
    expect(within(table).getByText("7,111")).toBeInTheDocument();
    expect(within(table).getByText("4,267 to 9,955")).toBeInTheDocument(); // 50% range of passing yards
    const rulers = screen.getAllByRole("img");
    expect(rulers.some((r) => r.getAttribute("aria-label")?.includes("90% range"))).toBe(true);
    expect(screen.getByText("Season-by-season path not available")).toBeInTheDocument(); // missing output explained
  });

  it("explains an unavailable career forecast instead of inventing numbers", () => {
    renderWithClient(<ProductionSection f={F.careerUnavailable()} />);
    expect(screen.getByText("Production forecasts are not available")).toBeInTheDocument();
    expect(screen.queryByRole("table")).not.toBeInTheDocument();
  });

  it("lists warnings and measured calibration", () => {
    renderWithClient(<UncertaintySection f={F.forecast()} />);
    expect(screen.getByText("Weakly calibrated output")).toBeInTheDocument();
    expect(screen.getByText(/79% for the 80% range/)).toBeInTheDocument();
  });
});

describe("pre-draft / post-draft separation", () => {
  it("hides the actual draft result in the pre-draft view until the reader asks", async () => {
    render(<ActualDraft detail={F.detail()} mode="pre_draft" />);
    expect(screen.queryByText(/pick 12/)).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Show" }));
    expect(screen.getByText(/round 1, pick 12/)).toBeInTheDocument();
  });

  it("shows it directly in the post-draft view", () => {
    render(<ActualDraft detail={F.detail()} mode="post_draft" />);
    expect(screen.getByText(/round 1, pick 12/)).toBeInTheDocument();
  });

  it("never requests a post-draft forecast for a player without a pick", async () => {
    const nav = await import("next/navigation");
    vi.spyOn(nav, "useSearchParams").mockReturnValue(new URLSearchParams("mode=post_draft") as never);
    const college = F.detail({ player_id: "cfb_123456_2020", kind: "college_only", drafted: false, draft_round: null, overall_pick: null });
    const { calls } = mockFetch({ "/api/prospects/cfb_123456_2020": [200, college] });
    renderWithClient(<ProspectReport id="cfb_123456_2020" />);
    expect(await screen.findByText(/has no recorded NFL draft slot/)).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: "Post-draft" })).toHaveAttribute("aria-checked", "true");
    expect(calls.some((c) => c.includes("/forecast"))).toBe(false);
  });
});

describe("historical backtest labels", () => {
  it.each([
    ["valid_out_of_sample_backtest", "Genuine out-of-sample backtest"],
    ["out_of_sample_forecast_outcome_incomplete", "Out-of-sample forecast, NFL window still open"],
    ["retrospective_illustration_in_sample", "In-sample illustration, not a backtest"],
  ] as const)("%s is labelled %s", (status, title) => {
    renderWithClient(<BacktestView b={F.backtest(status)} />);
    expect(screen.getByTestId("classification")).toHaveTextContent(title);
  });

  it("never presents in-sample numbers as forecast accuracy", () => {
    renderWithClient(<BacktestView b={F.backtest("retrospective_illustration_in_sample")} />);
    expect(screen.getByText(/trained on this player/)).toBeInTheDocument();
    expect(screen.getByRole("table", { name: /In-sample fit vs actual \(not forecast accuracy\)/ })).toBeInTheDocument();
  });

  it("shows errors as not yet known while the NFL window is open", () => {
    renderWithClient(<BacktestView b={F.backtest("out_of_sample_forecast_outcome_incomplete")} />);
    expect(screen.getAllByText("Not yet known").length).toBeGreaterThan(5);
    expect(screen.getAllByText("Window open").length).toBeGreaterThan(5);
  });
});

describe("model performance", () => {
  it("renders evaluated results with their evaluation status", async () => {
    mockFetch({ "/api/models/performance": [200, F.performance] });
    renderWithClient(<Methodology />);
    expect(await screen.findByText(/accounts for 80% of the improvement/)).toBeInTheDocument(); // (0.88-0.80)/(0.90-0.80)
    expect(screen.getByText(/fixture draft status/)).toBeInTheDocument();
    expect(screen.getByText("Fixture limitation one.")).toBeInTheDocument();
  });
});

describe("API errors", () => {
  it("explains an unreachable backend with a retry", async () => {
    const retry = vi.fn();
    render(<ErrorState error={new ApiError("BACKEND_UNREACHABLE", "The forecasting service could not be reached.", 0)} onRetry={retry} />);
    expect(screen.getByRole("alert")).toHaveTextContent("Forecasting service unavailable");
    fireEvent.click(screen.getByRole("button", { name: /Try again/ }));
    expect(retry).toHaveBeenCalled();
  });

  it("does not offer a retry for a missing prospect", () => {
    render(<ErrorState error={new ApiError("PROSPECT_NOT_FOUND", "No prospect with id qb_1.", 404)} onRetry={() => {}} />);
    expect(screen.queryByRole("button")).not.toBeInTheDocument();
    expect(screen.getByText("Error code: PROSPECT_NOT_FOUND")).toBeInTheDocument();
  });

  it("shows a backend failure on the report page without any forecast numbers", async () => {
    const { calls } = mockFetch({ "/api/prospects/qb_9001": [503, { error: { code: "MODEL_UNAVAILABLE", message: "Prediction data is not installed.", details: {} } }] });
    renderWithClient(<ProspectReport id="qb_9001" />);
    // a missing model bundle does not fix itself: reported at once, not retried (and never cached as data)
    expect(await screen.findByRole("alert", {}, { timeout: 8000 })).toHaveTextContent("Models not installed on the server");
    expect(calls).toHaveLength(1);
    expect(screen.queryByText(/Passing yards/)).not.toBeInTheDocument();
  }, 15000);
});

describe("accessibility basics (axe-core, structural rules)", () => {
  it("report sections have no violations", async () => {
    const f = F.forecast("post_draft");
    const { container } = renderWithClient(
      <div>
        <PlayerHeader detail={F.detail()} mode="post_draft" />
        <DraftSection f={f} />
        <StartsSection f={f} />
        <ProductionSection f={f} />
        <UncertaintySection f={f} />
        <Unavailable title="x" reason="y" />
      </div>,
    );
    await act(async () => {});
    expect(await axeViolations(container)).toEqual([]);
  });

  it("backtest view has no violations", async () => {
    const { container } = renderWithClient(<BacktestView b={F.backtest("valid_out_of_sample_backtest")} />);
    expect(await axeViolations(container)).toEqual([]);
  });

  it("search combobox exposes its state to assistive technology", async () => {
    mockFetch({ "/api/prospects": [200, F.searchResponse] });
    const { container } = renderWithClient(<ProspectSearch />);
    const box = screen.getByRole("combobox");
    expect(box).toHaveAttribute("aria-expanded", "false");
    await userEvent.type(box, "fixture");
    await waitFor(() => expect(box).toHaveAttribute("aria-expanded", "true"));
    await screen.findByRole("listbox");
    expect(await axeViolations(container)).toEqual([]);
  });
});

describe("Phase 7 additions", () => {
  it("mobile menu opens, lists every page and closes with Escape", async () => {
    const { SiteHeader } = await import("@/components/site-chrome");
    render(<SiteHeader />);
    const button = screen.getByRole("button", { name: "Open menu" });
    expect(button).toHaveAttribute("aria-expanded", "false");
    await userEvent.click(button);
    expect(button).toHaveAttribute("aria-expanded", "true");
    const menu = document.getElementById("mobile-menu")!;
    for (const name of ["Prospects", "Backtesting", "Methodology", "Case study", "GitHub"]) {
      expect(within(menu).getByRole("link", { name })).toBeInTheDocument();
    }
    await userEvent.keyboard("{Escape}");
    expect(document.getElementById("mobile-menu")).toBeNull();
  });

  it("every licensed photo carries its credit and licence", async () => {
    const { PHOTOS, creditLine } = await import("@/lib/images");
    const { PhotoBanner } = await import("@/components/photo-banner");
    for (const p of Object.values(PHOTOS)) {
      expect(p.credit && p.license && p.source.startsWith("https://commons.wikimedia.org/")).toBeTruthy();
      expect(p.alt.length).toBeGreaterThan(20);
    }
    render(<PhotoBanner photo={PHOTOS.heroStadium} title="t" />);
    expect(screen.getByRole("link", { name: creditLine(PHOTOS.heroStadium) })).toHaveAttribute("href", PHOTOS.heroStadium.source);
    expect(screen.getByRole("img", { name: PHOTOS.heroStadium.alt })).toBeInTheDocument();
  });

  it("credits page lists the data sources and every photo", async () => {
    const { default: Credits } = await import("@/app/credits/page");
    const { PHOTOS } = await import("@/lib/images");
    render(<Credits />);
    expect(screen.getByRole("link", { name: "CollegeFootballData.com" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "nflverse" })).toBeInTheDocument();
    expect(screen.getAllByRole("link", { name: "source" })).toHaveLength(Object.keys(PHOTOS).length);
  });

  it("comparison profile chart plots the prospect and comps from real college statistics", async () => {
    mockFetch({
      "/api/prospects/qb_9001/comparisons": [200, F.comparisons],
      "/api/prospects/qb_80": [200, F.detail()],
    });
    const { ComparisonsSection } = await import("@/components/report/comparisons-section");
    renderWithClient(<ComparisonsSection id="qb_9001" detail={F.detail()} />);
    expect(await screen.findByText("College profile against the five comparisons (final season)")).toBeInTheDocument();
    const strip = screen.getByRole("img", { name: /^Cmp %:/ });
    expect(strip.getAttribute("aria-label")).toContain("Fixture QB Alpha 66.1");
    expect(screen.getAllByText("FC").length).toBeGreaterThanOrEqual(5); // initials, not photos
  });
});
