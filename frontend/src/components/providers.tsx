"use client";
import { useState } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { TooltipProvider } from "@/components/ui/tooltip";
import { retryDelay, retryPolicy } from "@/lib/api/queries";

export function Providers({ children }: { children: React.ReactNode }) {
  const [client] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: { refetchOnWindowFocus: false, retry: retryPolicy, retryDelay, gcTime: 2 * 60 * 60 * 1000 },
        },
      }),
  );
  return (
    <QueryClientProvider client={client}>
      <TooltipProvider delay={150}>{children}</TooltipProvider>
    </QueryClientProvider>
  );
}
