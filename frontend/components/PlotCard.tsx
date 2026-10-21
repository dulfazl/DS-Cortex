"use client";

import { useEffect, useRef, useState } from "react";
import type { Plot } from "@/lib/types";

function themedLayout(layout: Record<string, unknown>, el: HTMLElement) {
  const css = getComputedStyle(el);
  const text = css.getPropertyValue("--text").trim() || "#18202c";
  const grid = css.getPropertyValue("--border").trim() || "#e2e6ec";
  const axis = { gridcolor: grid, zerolinecolor: grid, linecolor: grid, color: text };
  return {
    ...layout,
    title: undefined,
    autosize: true,
    height: 340,
    margin: { l: 56, r: 16, t: 16, b: 48 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: text, size: 12, family: "ui-sans-serif, system-ui, sans-serif" },
    xaxis: { ...((layout.xaxis as object) ?? {}), ...axis },
    yaxis: { ...((layout.yaxis as object) ?? {}), ...axis, automargin: true },
    legend: { orientation: "h", y: -0.2 },
  };
}

/** Renders Plotly JSON client-side. Plotly is loaded lazily so the first paint stays light. */
export function PlotCard({ plot }: { plot: Plot }) {
  const ref = useRef<HTMLDivElement>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const el = ref.current;
    if (!el) return;
    let observer: ResizeObserver | null = null;
    import("plotly.js-dist-min")
      .then(({ default: Plotly }) => {
        if (cancelled || !el) return;
        Plotly.react(el, plot.figure.data, themedLayout(plot.figure.layout, el), { responsive: true, displaylogo: false });
        observer = new ResizeObserver(() => Plotly.Plots.resize(el));
        observer.observe(el);
      })
      .catch((e) => setError(String(e)));
    return () => {
      cancelled = true;
      observer?.disconnect();
      import("plotly.js-dist-min").then(({ default: Plotly }) => el && Plotly.purge(el)).catch(() => undefined);
    };
  }, [plot]);

  return (
    <figure className="card min-w-0 p-4">
      <figcaption className="mb-2">
        <div className="flex items-start gap-2">
          <h3 className="text-sm font-semibold">{plot.title}</h3>
          <span className="ml-auto shrink-0 rounded bg-surface-2 px-1.5 text-[11px] text-muted" title="Relevance to your problem statement">
            relevance {plot.relevance.toFixed(2)}
          </span>
        </div>
        <p className="mt-0.5 text-xs text-muted">{plot.caption}</p>
      </figcaption>
      <div className="w-full overflow-x-auto">
        <div ref={ref} className="h-[340px] min-w-[280px] w-full" aria-label={plot.title} role="img" />
      </div>
      {error && <p className="text-xs text-bad">Could not render this plot: {error}</p>}
    </figure>
  );
}
