import { Suspense } from "react";
import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { isPlayerId } from "@/lib/api/client";
import { ProspectReport } from "@/components/report/prospect-report";
import { SlowLoading } from "@/components/status";

export const metadata: Metadata = { title: "Prospect report" };

/** Reads the route param inside the Suspense boundary so the page shell can stream immediately. */
async function Report({ params }: { params: PageProps<"/prospects/[id]">["params"] }) {
  const { id } = await params;
  if (!isPlayerId(id)) notFound();
  return <ProspectReport id={id} />;
}

export default function ProspectPage(props: PageProps<"/prospects/[id]">) {
  return (
    <Suspense
      fallback={
        <div className="mx-auto max-w-6xl px-4 py-12 sm:px-6">
          <SlowLoading label="Loading prospect" />
        </div>
      }
    >
      <Report params={props.params} />
    </Suspense>
  );
}
