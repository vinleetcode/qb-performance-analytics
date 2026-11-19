import { Suspense } from "react";
import type { Metadata } from "next";
import { ProspectExplorer } from "@/components/prospect-explorer";
import { SlowLoading } from "@/components/status";

export const metadata: Metadata = { title: "Prospects" };

export default function ProspectsPage() {
  return (
    <Suspense fallback={<div className="mx-auto max-w-6xl px-4 py-12 sm:px-6"><SlowLoading label="Loading prospects" /></div>}>
      <ProspectExplorer />
    </Suspense>
  );
}
