import Link from "next/link";

export default function NotFound() {
  return (
    <div className="mx-auto max-w-6xl px-4 py-24 sm:px-6">
      <h1 className="text-[56px] text-chalk">Page not found</h1>
      <p className="mt-3 text-muted-foreground">
        That address does not match a page or a prospect id.{" "}
        <Link href="/prospects" className="text-projection-text underline decoration-1 hover:decoration-2">Search for a quarterback</Link> instead.
      </p>
    </div>
  );
}
