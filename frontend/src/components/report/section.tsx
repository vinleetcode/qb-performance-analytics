import { cn } from "@/lib/utils";

/** A report section: a heading on the left rail (desktop) and the content on the right. */
export function Section({
  id,
  title,
  intro,
  children,
  className,
}: {
  id: string;
  title: string;
  intro?: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <section id={id} aria-labelledby={`${id}-h`} className={cn("grid gap-6 border-b border-line py-12 lg:grid-cols-[15rem_1fr]", className)}>
      <div>
        <h2 id={`${id}-h`} className="text-[30px] text-chalk">
          {title}
        </h2>
        {intro && <p className="mt-2 text-[14px] text-muted-foreground lg:max-w-[15rem]">{intro}</p>}
      </div>
      <div className="min-w-0">{children}</div>
    </section>
  );
}
