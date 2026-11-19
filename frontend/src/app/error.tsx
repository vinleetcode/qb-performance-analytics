"use client";

export default function GlobalError({ reset }: { error: Error & { digest?: string }; reset: () => void }) {
  return (
    <div className="mx-auto max-w-6xl px-4 py-24 sm:px-6" role="alert">
      <h1 className="text-[48px] text-chalk">This page failed to load</h1>
      <p className="mt-3 text-muted-foreground">An unexpected error stopped the page. No forecast data was shown.</p>
      <button type="button" onClick={reset} className="mt-6 rounded-md border border-line px-4 py-2 text-sm text-chalk hover:bg-panel">
        Try again
      </button>
    </div>
  );
}
