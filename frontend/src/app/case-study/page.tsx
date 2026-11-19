import type { Metadata } from "next";
import { CaseStudy } from "@/components/case-study";

export const metadata: Metadata = {
  title: "Case study",
  description: "How QB Forecast Lab was built: data engineering, modelling, chronological backtesting, calibration and the product.",
};

export default function CaseStudyPage() {
  return <CaseStudy />;
}
