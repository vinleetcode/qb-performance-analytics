"use client";
import { useEffect, useId, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { Loader2, Search, Users } from "lucide-react";
import { useProspectSearch } from "@/lib/api/queries";
import type { ProspectSummary } from "@/lib/api/types";
import { cn } from "@/lib/utils";

export function useDebounced<T>(value: T, ms = 200) {
  const [v, setV] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setV(value), ms);
    return () => clearTimeout(t);
  }, [value, ms]);
  return v;
}

export function draftLine(p: ProspectSummary) {
  if (p.kind === "drafted" && p.overall_pick != null) return `${p.draft_year} draft, round ${p.draft_round}, pick ${p.overall_pick}`;
  if (p.drafted === true) return `${p.draft_year} draft, drafted`;
  if (p.drafted === false) return `${p.draft_year} draft class, not drafted`;
  return `${p.draft_year} draft class, status not yet verifiable`;
}

/**
 * Autocomplete over the real prospect index (ARIA combobox: arrow keys, Enter, Escape).
 * Names are not unique, so every option shows college and draft class, and navigation uses the stable player id.
 */
export function ProspectSearch({
  onSelect,
  draftedOnly = false,
  placeholder = "Search quarterbacks by name",
  autoFocus = false,
  className,
  label = "Search quarterbacks",
}: {
  onSelect?: (p: ProspectSummary) => void;
  draftedOnly?: boolean;
  placeholder?: string;
  autoFocus?: boolean;
  className?: string;
  label?: string;
}) {
  const router = useRouter();
  const [text, setText] = useState("");
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const q = useDebounced(text.trim(), 200);
  const enabled = q.length >= 2;
  const { data, isFetching, isError } = useProspectSearch(q, { draftedOnly, limit: 8, enabled });
  const items = enabled ? (data?.items ?? []) : [];
  const id = useId();
  const listRef = useRef<HTMLUListElement>(null);

  const choose = (p: ProspectSummary) => {
    setOpen(false);
    setText(p.name);
    if (onSelect) onSelect(p);
    else router.push(`/prospects/${p.player_id}`);
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setOpen(true);
      setActive((a) => Math.min(a + 1, Math.max(items.length - 1, 0)));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActive((a) => Math.max(a - 1, 0));
    } else if (e.key === "Enter") {
      if (open && items[active]) {
        e.preventDefault();
        choose(items[active]);
      }
    } else if (e.key === "Escape") {
      setOpen(false);
    }
  };

  const showList = open && enabled;
  return (
    <div className={cn("relative", className)}>
      <label htmlFor={`${id}-input`} className="sr-only">
        {label}
      </label>
      <div className="flex h-12 items-center gap-2 rounded-md border border-line bg-panel px-3 focus-within:border-projection-text">
        <Search className="size-4 shrink-0 text-muted-foreground" aria-hidden />
        <input
          id={`${id}-input`}
          role="combobox"
          aria-expanded={showList}
          aria-controls={`${id}-list`}
          aria-autocomplete="list"
          aria-activedescendant={showList && items[active] ? `${id}-opt-${active}` : undefined}
          autoComplete="off"
          spellCheck={false}
          autoFocus={autoFocus}
          value={text}
          placeholder={placeholder}
          onChange={(e) => {
            setText(e.target.value);
            setOpen(true);
            setActive(0);
          }}
          onFocus={() => setOpen(true)}
          onBlur={() => setTimeout(() => setOpen(false), 120)}
          onKeyDown={onKeyDown}
          className="h-full min-w-0 flex-1 bg-transparent text-[15px] text-chalk placeholder:text-muted-foreground focus:outline-none"
        />
        {isFetching && enabled && <Loader2 className="size-4 animate-spin text-muted-foreground" aria-label="Searching" />}
      </div>
      {showList && (
        <ul
          id={`${id}-list`}
          ref={listRef}
          role="listbox"
          aria-label="Matching quarterbacks"
          className="absolute z-30 mt-1 max-h-96 w-full overflow-y-auto rounded-md border border-line bg-raised py-1 shadow-xl shadow-black/40"
        >
          {isError && <li className="px-3 py-2 text-sm text-destructive">Search is unavailable right now. The forecasting server may be waking up; try again in a minute.</li>}
          {!isError && !isFetching && items.length === 0 && (
            <li className="px-3 py-2 text-sm text-muted-foreground">No quarterback matches &ldquo;{q}&rdquo;.</li>
          )}
          {items.map((p, i) => (
            <li
              key={p.player_id}
              id={`${id}-opt-${i}`}
              role="option"
              aria-selected={i === active}
              onMouseDown={(e) => {
                e.preventDefault();
                choose(p);
              }}
              onMouseEnter={() => setActive(i)}
              className={cn("cursor-pointer px-3 py-2", i === active && "bg-panel")}
            >
              <div className="flex items-baseline justify-between gap-3">
                <span className="font-medium text-chalk">{p.name}</span>
                <span className="num shrink-0 text-sm text-muted-foreground">{p.draft_year}</span>
              </div>
              <div className="flex flex-wrap items-center gap-x-2 text-[13px] text-muted-foreground">
                <span>{p.college ?? "College unknown"}</span>
                <span aria-hidden>/</span>
                <span>{draftLine(p)}</span>
                {p.shares_name_with_other_prospects && (
                  <span className="inline-flex items-center gap-1 text-caution">
                    <Users className="size-3" aria-hidden />
                    shared name
                  </span>
                )}
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
