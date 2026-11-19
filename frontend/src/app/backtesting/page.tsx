import { Suspense } from "react";
import type { Metadata } from "next";
import { BacktestExplorer } from "@/components/backtest/backtest-explorer";
import { SlowLoading } from "@/components/status";

export const metadata: Metadata = { title: "Backtesting" };

export default function BacktestingPage() {
  return (
    <Suspense fallback={<div className="mx-auto max-w-6xl px-4 py-12 sm:px-6"><SlowLoading label="Loading" /></div>}>
      <BacktestExplorer />
    </Suspense>
  );
}
