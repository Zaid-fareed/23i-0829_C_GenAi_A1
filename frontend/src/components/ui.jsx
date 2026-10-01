import React from "react";
import { download } from "../api.js";

export const Mono = ({ children, className = "" }) => (
  <span className={`font-mono tabular-nums ${className}`}>{children}</span>
);

/** Small section block used inside the control panel (numbered like the Stitch design: "01 Input image"). */
export const Section = ({ n, title, tag, children }) => (
  <section>
    <div className="mb-2 flex items-center justify-between gap-2">
      <h3 className="flex items-center gap-2 text-[15px] font-semibold tracking-tight text-slate-900">
        <span className="grid h-5 w-5 shrink-0 place-items-center rounded-md bg-brand-50 font-mono text-[11px] font-semibold text-brand-700">{n}</span>
        {title}
      </h3>
      {tag && <span className="font-mono text-[10px] font-semibold uppercase tracking-widest text-slate-400">{tag}</span>}
    </div>
    {children}
  </section>
);

export const Card = ({ title, right, children, className = "" }) => (
  <section className={`rounded-xl border border-slate-200 bg-white p-3 ${className}`}>
    {(title || right) && (
      <div className="mb-2 flex items-center justify-between gap-2">
        <h3 className="text-[15px] font-semibold tracking-tight text-slate-900">{title}</h3>
        {right}
      </div>
    )}
    {children}
  </section>
);

/**
 * Two-panel workbench: narrow controls on the left (Run button pinned at its bottom), results on the right.
 * On desktop each panel scrolls on its own so the Run button and the results are always on screen together;
 * on phones the Run button is pinned to the bottom of the screen.
 */
export const Workbench = ({ controls, footer, children }) => (
  <div className="flex flex-col gap-4 pb-24 lg:h-full lg:flex-row lg:pb-0">
    <div className="flex flex-col rounded-2xl border border-slate-200 bg-white shadow-sm lg:min-h-0 lg:w-[340px] lg:shrink-0">
      <div className="space-y-5 p-4 lg:min-h-0 lg:flex-1 lg:overflow-y-auto">{controls}</div>
      <div className="fixed inset-x-0 bottom-0 z-30 border-t border-slate-200 bg-white p-3 lg:static lg:rounded-b-2xl lg:border-slate-100 lg:p-4">{footer}</div>
    </div>
    <div className="min-w-0 rounded-2xl border border-slate-200 bg-white p-4 shadow-sm sm:p-5 lg:flex-1 lg:overflow-y-auto">{children}</div>
  </div>
);

/**
 * A row of equal SQUARE image tiles that grows with the panel but never taller than the screen allows
 * (`reserve` = pixels of the screen kept for everything else in the result panel), so there is no long empty space.
 */
export const TileRow = ({ cols, reserve, children }) => (
  <div style={{ "--reserve": `${reserve}px`, "--cols": cols }}
    className={`mx-auto grid gap-3 ${cols === 3 ? "grid-cols-1 sm:grid-cols-3" : "grid-cols-1 sm:grid-cols-2"} lg:max-w-[calc((100vh-var(--reserve))*var(--cols)+(var(--cols)-1)*0.75rem)]`}>
    {children}
  </div>
);

export const Button = ({ variant = "primary", className = "", ...p }) => (
  <button
    {...p}
    className={`inline-flex items-center justify-center gap-2 rounded-lg px-4 py-2.5 text-sm font-medium transition focus:outline-none focus-visible:ring-2 focus-visible:ring-brand-500/30 disabled:cursor-not-allowed disabled:opacity-50 ${
      variant === "primary" ? "bg-brand-600 text-white hover:bg-brand-700" : "border border-slate-200 bg-white text-slate-700 hover:bg-slate-100"
    } ${className}`}
  />
);

/** Segmented control: slate-100 track, active segment is an elevated white tab. */
export const Segmented = ({ options, value, onChange }) => (
  <div className="grid gap-1 rounded-lg bg-slate-100 p-1" style={{ gridTemplateColumns: `repeat(${options.length}, minmax(0, 1fr))` }}>
    {options.map(([k, l]) => (
      <button key={k} type="button" onClick={() => onChange(k)}
        className={`rounded-md px-2 py-1.5 text-sm transition ${value === k ? "bg-white font-medium text-brand-700 shadow-sm" : "text-slate-500 hover:text-slate-700"}`}>
        {l}
      </button>
    ))}
  </div>
);

const TONES = {
  ok: ["bg-emerald-50 text-emerald-700", "text-emerald-500 bg-emerald-500"],
  warn: ["bg-amber-50 text-amber-700", "text-amber-500 bg-amber-500"],
  bad: ["bg-rose-50 text-rose-700", "text-rose-500 bg-rose-500"],
};
export const Pill = ({ tone = "ok", pulse = false, children }) => (
  <span className={`inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 font-mono text-[11px] font-medium ${TONES[tone][0]}`}>
    <span className={`relative h-1.5 w-1.5 rounded-full ${TONES[tone][1]} ${pulse ? "pulse-dot" : ""}`} />
    {children}
  </span>
);

/** Compact info tile for the metrics strip. */
export const Stat = ({ label, value, hint }) => (
  <div className="min-w-0 rounded-xl border border-slate-100 bg-slate-50 px-3 py-1.5">
    <div className="font-mono text-[10px] font-semibold uppercase tracking-widest text-slate-500">{label}</div>
    <div className="mt-0.5 break-words font-mono text-[15px] font-semibold leading-tight text-slate-900">{value}</div>
    {hint && <div className="mt-0.5 font-mono text-[11px] leading-tight text-slate-400">{hint}</div>}
  </div>
);

/** Square image tile with title, PNG download and overlay badge. Shows a dashed placeholder before the first run. */
export const ImagePanel = ({ index, title, src, filename, badge, hint = "Run the model to see this" }) => (
  <figure className="min-w-0">
    <div className="mb-1.5 flex items-center justify-between gap-2 text-sm font-medium text-slate-800">
      <span className="truncate">{index != null && <span className="text-slate-400">{index}. </span>}{title}</span>
      {src && filename && (
        <button onClick={() => download(src, filename)} className="shrink-0 font-mono text-[11px] font-semibold uppercase text-brand-600 hover:text-brand-700">↓ PNG</button>
      )}
    </div>
    <div className={`relative aspect-square w-full overflow-hidden rounded-xl border ${src ? "border-slate-200 bg-slate-100" : "border-dashed border-slate-300 bg-slate-50"}`}>
      {src ? (
        <img src={src} alt={title} className="h-full w-full object-cover" />
      ) : (
        <div className="flex h-full items-center justify-center p-3 text-center font-mono text-[11px] text-slate-400">{hint}</div>
      )}
      {badge && <span className="absolute left-2 top-2 rounded-md bg-slate-900/70 px-2 py-0.5 font-mono text-[10px] text-white">{badge}</span>}
    </div>
  </figure>
);

/** Probability / weight rows. `selected` (key) gets the highlighted treatment. */
export const WeightBars = ({ data, selected, selectedLabel = "Selected", cols = 1 }) => {
  const entries = Object.entries(data);
  const max = Math.max(...entries.map(([, v]) => v));
  return (
    <ul className={cols === 2 ? "grid grid-cols-1 gap-x-3 gap-y-1 sm:grid-cols-2" : "space-y-1"}>
      {entries.map(([k, v]) => {
        const top = selected ? k === selected : v === max;
        return (
          <li key={k} className={`rounded-lg px-2.5 py-1.5 ${top ? "bg-brand-50/80" : ""}`}>
            <div className="mb-1 flex items-center justify-between gap-2 text-sm">
              <span className="flex min-w-0 flex-wrap items-center gap-2">
                <span className={`truncate ${top ? "font-semibold text-slate-900" : "text-slate-600"}`}>{k}</span>
                {top && <span className="rounded-full bg-brand-600 px-2 py-px font-mono text-[10px] font-medium text-white">{selectedLabel}</span>}
              </span>
              <Mono className={top ? "font-semibold text-brand-700" : "text-slate-600"}>{(v * 100).toFixed(1)}%</Mono>
            </div>
            <div className="h-1.5 overflow-hidden rounded-full bg-slate-100">
              <div className={`h-full rounded-full transition-all duration-500 ${top ? "bg-brand-600" : "bg-slate-400"}`} style={{ width: `${Math.max(v * 100, 0.6)}%` }} />
            </div>
          </li>
        );
      })}
    </ul>
  );
};

export const ErrorBox = ({ msg }) =>
  msg ? <div role="alert" className="rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-sm text-rose-700">{msg}</div> : null;
