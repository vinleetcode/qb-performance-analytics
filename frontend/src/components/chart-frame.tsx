"use client";
import { useId, useState } from "react";
import { Table2 } from "lucide-react";
import { cn } from "@/lib/utils";

/**
 * Accessible wrapper for every chart: a figure with a caption, a text description for screen readers, and a
 * "Show data" table with the exact numbers drawn.
 */
export function ChartFrame({
  title,
  description,
  columns,
  rows,
  children,
  className,
}: {
  title: string;
  description: string;
  columns: string[];
  rows: (string | number)[][];
  children: React.ReactNode;
  className?: string;
}) {
  const [show, setShow] = useState(false);
  const id = useId();
  return (
    <figure className={cn("min-w-0", className)} aria-labelledby={`${id}-t`} aria-describedby={`${id}-d`}>
      <div className="mb-2 flex items-start justify-between gap-3">
        <figcaption id={`${id}-t`} className="text-sm font-medium text-chalk">
          {title}
        </figcaption>
        <button
          type="button"
          onClick={() => setShow((s) => !s)}
          aria-expanded={show}
          aria-controls={`${id}-table`}
          className="inline-flex shrink-0 items-center gap-1 rounded text-xs text-muted-foreground hover:text-chalk"
        >
          <Table2 className="size-3.5" aria-hidden />
          {show ? "Hide data" : "Show data"}
        </button>
      </div>
      <p id={`${id}-d`} className="sr-only">
        {description}
      </p>
      <div>{children}</div>
      {show && (
        <div id={`${id}-table`} className="mt-3 overflow-x-auto">
          <table className="w-full text-left text-[13px]">
            <thead>
              <tr className="border-b border-line text-muted-foreground">
                {columns.map((c) => (
                  <th key={c} scope="col" className="py-1.5 pr-3 font-medium">
                    {c}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <tr key={i} className="border-b border-line/50">
                  {r.map((c, j) => (
                    <td key={j} className="num py-1.5 pr-3">
                      {c}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </figure>
  );
}
