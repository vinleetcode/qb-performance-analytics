"use client";
import { useEffect, useState } from "react";
import { AlertTriangle, Ban, CircleHelp, Info, RefreshCw, ServerOff } from "lucide-react";
import { ApiError } from "@/lib/api/client";
import type { PredictionWarning } from "@/lib/api/types";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";
import { humanize } from "@/lib/format";

/** A prediction the model does not support. Never a zero: a dashed panel with the backend's reason. */
export function Unavailable({ title, reason, className }: { title: string; reason?: string | null; className?: string }) {
  return (
    <div
      role="note"
      className={cn("flex gap-3 rounded-md border border-dashed border-line bg-panel/40 p-4 text-sm", className)}
      data-testid="unavailable"
    >
      <Ban className="mt-0.5 size-4 shrink-0 text-muted-foreground" aria-hidden />
      <div>
        <p className="font-medium text-chalk">{title}</p>
        {reason && <p className="mt-1 text-muted-foreground">{humanize(reason)}</p>}
      </div>
    </div>
  );
}

const WARNING_TITLE: Record<string, string> = {
  DRAFT_PROBABILITY_UNAVAILABLE: "Draft probability not available",
  COMBINE_PARTICIPATION: "Combine probability reflects invitations",
  CALIBRATION_WARNING: "Weakly calibrated output",
  EXTRAPOLATION: "Beyond the historical range",
  HYPOTHETICAL_PICK: "Hypothetical draft slot",
  ASSUMPTION: "Assumption",
};

export function WarningList({ warnings, className }: { warnings: PredictionWarning[]; className?: string }) {
  if (!warnings.length) return null;
  return (
    <ul className={cn("space-y-2", className)} aria-label="Model warnings">
      {warnings.map((w, i) => (
        <li key={`${w.code}-${i}`} className="flex gap-3 rounded-md border-l-2 border-caution bg-panel/60 px-3 py-2 text-sm">
          <AlertTriangle className="mt-0.5 size-4 shrink-0 text-caution" aria-hidden />
          <div>
            <p className="font-medium text-chalk">
              <span className="sr-only">Warning: </span>
              {WARNING_TITLE[w.code] ?? "Note"}
            </p>
            <p className="text-muted-foreground">{humanize(w.message)}</p>
          </div>
        </li>
      ))}
    </ul>
  );
}

const ERROR_TEXT: Partial<Record<string, { title: string; hint: string }>> = {
  PROSPECT_NOT_FOUND: { title: "Prospect not found", hint: "Search for the quarterback again; ids change only when the data is rebuilt." },
  BACKEND_UNREACHABLE: {
    title: "Forecasting service unavailable",
    hint:
      process.env.NODE_ENV === "development"
        ? "Start the API (uvicorn api.main:app --port 8000) or check NEXT_PUBLIC_API_BASE_URL."
        : "The forecasting server runs on a free hosting tier and sleeps when nobody has used it for a while. It usually wakes within a minute or two: try again shortly. The rest of the site works meanwhile.",
  },
  MODEL_UNAVAILABLE: { title: "Models not installed on the server", hint: "The serving bundle is missing or failed its integrity check." },
  RATE_LIMITED: { title: "Too many requests", hint: "Wait a minute, then try again." },
  PREDICTION_UNSUPPORTED: { title: "Not supported for this player", hint: "" },
  INVALID_PARAMETER: { title: "Invalid request", hint: "The link may be malformed." },
  INVALID_PREDICTION_MODE: { title: "Unknown forecast mode", hint: "Use pre-draft or post-draft." },
};

export function ErrorState({ error, onRetry, className }: { error: unknown; onRetry?: () => void; className?: string }) {
  const e = error instanceof ApiError ? error : null;
  const text = (e && ERROR_TEXT[e.code]) || { title: "Something went wrong", hint: "" };
  const Icon = e?.code === "BACKEND_UNREACHABLE" || e?.code === "MODEL_UNAVAILABLE" ? ServerOff : CircleHelp;
  return (
    <div role="alert" className={cn("flex flex-col gap-3 rounded-md border border-line bg-panel p-5 text-sm", className)}>
      <div className="flex gap-3">
        <Icon className="mt-0.5 size-5 shrink-0 text-destructive" aria-hidden />
        <div>
          <p className="font-medium text-chalk">{text.title}</p>
          <p className="mt-1 text-muted-foreground">{e?.message ?? "The request failed."}</p>
          {text.hint && <p className="mt-1 text-muted-foreground">{text.hint}</p>}
          {e && <p className="mt-2 text-xs text-muted-foreground">Error code: {e.code}</p>}
        </div>
      </div>
      {onRetry && (!e || e.retryable) && (
        <div>
          <Button variant="outline" size="sm" onClick={onRetry}>
            <RefreshCw aria-hidden /> Try again
          </Button>
        </div>
      )}
    </div>
  );
}

/**
 * Skeleton plus honest feedback for slow requests: after a few seconds it explains that the first forecast after a
 * server start loads the models. No invented progress percentage.
 */
export function SlowLoading({ label, lines = 3, slowAfterMs = 2500 }: { label: string; lines?: number; slowAfterMs?: number }) {
  const [slow, setSlow] = useState(false);
  useEffect(() => {
    const t = setTimeout(() => setSlow(true), slowAfterMs);
    return () => clearTimeout(t);
  }, [slowAfterMs]);
  return (
    <div role="status" aria-live="polite" className="space-y-3">
      <span className="sr-only">{label}</span>
      {Array.from({ length: lines }).map((_, i) => (
        <Skeleton key={i} className={cn("h-6 bg-panel", i === 0 ? "w-2/3" : "w-full")} />
      ))}
      {slow && (
        <p className="flex items-center gap-2 text-sm text-muted-foreground">
          <Info className="size-4" aria-hidden />
          Still working. The forecasting server is on a free hosting tier and sleeps when idle, so the first request
          after a quiet spell can take a minute or two while it wakes and loads the models. Nothing is shown until the
          real model output arrives.
        </p>
      )}
    </div>
  );
}
