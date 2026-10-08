import { objectIds, samples } from "@/lib/data";

function Row({ ids, reverse, pick }: { ids: string[]; reverse?: boolean; pick: number }) {
  const items = [...ids, ...ids];
  return (
    <div className="flex w-max gap-3 marquee" style={{ animationDirection: reverse ? "reverse" : "normal" }}>
      {items.map((id, i) => (
        <figure key={i} className="relative h-28 w-28 shrink-0 overflow-hidden rounded-2xl border border-line bg-surface sm:h-36 sm:w-36" aria-hidden={i >= ids.length}>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={samples.singles[id][pick % samples.singles[id].length]} alt={i < ids.length ? `${id} test photo` : ""} loading="lazy" className="h-full w-full object-cover" />
          <figcaption className="absolute bottom-1.5 left-1.5 rounded-md bg-black/60 px-1.5 py-0.5 text-[10px] font-medium text-white backdrop-blur">{id}</figcaption>
        </figure>
      ))}
    </div>
  );
}

export function ObjectMarquee() {
  const half = Math.ceil(objectIds.length / 2);
  return (
    <div className="relative overflow-hidden py-6 [perspective:1400px]">
      <div className="space-y-3 [transform:rotateX(18deg)_rotateZ(-4deg)_scale(1.08)] [mask-image:linear-gradient(to_right,transparent,black_12%,black_88%,transparent)]">
        <Row ids={objectIds.slice(0, half)} pick={0} />
        <Row ids={objectIds.slice(half)} pick={1} reverse />
      </div>
    </div>
  );
}
