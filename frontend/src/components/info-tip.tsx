"use client";
import { Info } from "lucide-react";
import { Popover as P } from "@base-ui/react/popover";
import { cn } from "@/lib/utils";

/**
 * A toggletip: an explanation next to a number or heading that works with every input.
 * Mouse: opens on hover (and click). Keyboard: Tab to the button, Enter or Space toggles, Escape closes.
 * Touch: a tap toggles it and a tap outside closes it. Plain tooltips do not open on touch, hence a popover.
 */
export function InfoTip({ label, children, className }: { label: string; children: React.ReactNode; className?: string }) {
  return (
    <P.Root>
      <P.Trigger
        openOnHover
        delay={120}
        closeDelay={80}
        aria-label={`About ${label}`}
        className={cn(
          "relative inline-flex size-5 items-center justify-center rounded-full align-middle text-muted-foreground hover:text-chalk data-[popup-open]:text-chalk",
          // a 32px touch target around the 20px icon
          "after:absolute after:-inset-1.5 after:content-['']",
          className,
        )}
      >
        <Info className="size-3.5" aria-hidden />
      </P.Trigger>
      <P.Portal>
        <P.Positioner side="top" sideOffset={6} collisionPadding={12} className="z-50">
          <P.Popup className="max-w-[min(18rem,calc(100vw-24px))] rounded-md border border-line bg-raised px-3 py-2 text-left text-[13px] leading-snug font-normal text-chalk shadow-lg shadow-black/40 outline-none">
            {children}
          </P.Popup>
        </P.Positioner>
      </P.Portal>
    </P.Root>
  );
}
