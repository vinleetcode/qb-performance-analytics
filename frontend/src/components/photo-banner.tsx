import Image from "next/image";
import { creditLine, type SitePhoto } from "@/lib/images";
import { cn } from "@/lib/utils";

export function PhotoCredit({ photo, className }: { photo: SitePhoto; className?: string }) {
  return (
    <a
      href={photo.source}
      target="_blank"
      rel="noreferrer"
      className={cn("text-[11px] text-chalk/70 underline decoration-chalk/30 underline-offset-2 hover:text-chalk", className)}
    >
      {creditLine(photo)}
    </a>
  );
}

/**
 * Editorial page banner: a licensed photograph behind a strong scrim, the page title and intro on top, and the photo
 * credit in the corner. Photos never sit behind charts or numbers.
 */
export function PhotoBanner({
  photo,
  title,
  intro,
  children,
  priority = false,
  tall = false,
  grayscale = false,
}: {
  photo: SitePhoto;
  title: React.ReactNode;
  intro?: React.ReactNode;
  children?: React.ReactNode;
  priority?: boolean;
  tall?: boolean;
  grayscale?: boolean;
}) {
  return (
    <section className={cn("relative isolate overflow-hidden border-b border-line", tall ? "min-h-[min(88vh,820px)]" : "min-h-[340px]")}>
      <Image
        src={photo.src}
        alt={photo.alt}
        fill
        priority={priority}
        sizes="100vw"
        quality={72}
        className={cn("-z-20 object-cover", grayscale && "grayscale")}
        style={{ objectPosition: photo.position ?? "50% 50%" }}
      />
      {/* scrim: keeps text at full contrast whatever the photo does */}
      <div className="absolute inset-0 -z-10 bg-[linear-gradient(90deg,rgb(15_26_43/0.94)_0%,rgb(15_26_43/0.78)_45%,rgb(15_26_43/0.35)_100%)]" />
      <div className="absolute inset-x-0 bottom-0 -z-10 h-40 bg-gradient-to-t from-background to-transparent" />
      <div className={cn("mx-auto flex max-w-6xl flex-col justify-end px-4 sm:px-6", tall ? "min-h-[min(88vh,820px)] py-20" : "min-h-[340px] py-12")}>
        <h1 className="max-w-3xl text-[clamp(42px,7vw,88px)] leading-[0.92] text-chalk">{title}</h1>
        {intro && <div className="measure mt-5 text-[17px] text-chalk/85">{intro}</div>}
        {children}
      </div>
      <PhotoCredit photo={photo} className="absolute right-3 bottom-2 sm:right-6" />
    </section>
  );
}
