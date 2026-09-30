import React from "react";
import { download } from "../api.js";

export const Mono = ({ children, className = "" }) => (
  <span className={`font-mono tabular-nums ${className}`}>{children}</span>
);

/** Card with optional numbered step badge (Stitch: "1 · Input image") and a mono tag on the right. */
export const Card = ({ title, step, tag, right, children, className = "" }) => (
  <section className={`rounded-2xl border border-slate-200 bg-white p-5 shadow-[0_1px_3px_0_rgba(15,23,42,0.05),0_1px_2px_-1px_rgba(15,23,42,0.03)] ${className}`}>
    {(title || right || tag) && (
      <div className="mb-4 flex items-center justify-between gap-2">
        <h3 className="flex items-center gap-2 text-[17px] font-semibold tracking-tight text-slate-900">
          {step != null && (
            <span className="grid h-6 w-6 shrink-0 place-items-center rounded-full bg-brand-50 font-mono text-xs font-semibold text-brand-700">{step}</span>
          )}
          {title}
        </h3>
        {right ?? (tag && <span className="font-mono text-[10px] font-semibold uppercase tracking-widest text-slate-500">{tag}</span>)}
      </div>
    )}
    {children}
  </section>
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
        className={`rounded-md px-2 py-1.5 text-sm transition ${value === k ? "bg-white font-medium text-slate-900 shadow-sm" : "text-slate-500 hover:text-slate-700"}`}>
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
/** Status pill with a pulsing dot (Stitch: Ready / Warmup / Fault). */
export const Pill = ({ tone = "ok", pulse = false, children }) => (
  <span className={`inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 font-mono text-[11px] font-medium ${TONES[tone][0]}`}>
    <span className={`relative h-1.5 w-1.5 rounded-full ${TONES[tone][1]} ${pulse ? "pulse-dot" : ""}`} />
    {children}
  </span>
);

export const Stat = ({ label, value, hint }) => (
  <div className="rounded-xl border border-slate-100 bg-slate-50 px-4 py-3">
    <div className="font-mono text-[10px] font-semibold uppercase tracking-widest text-slate-500">{label}</div>
    <div className="mt-1 break-words font-mono text-lg font-semibold text-slate-900">{value}</div>
    {hint && <div className="mt-0.5 font-mono text-xs text-slate-400">{hint}</div>}
  </div>
);

/** One image with a numbered title, PNG download link and overlay badge. Images are 128px shown enlarged. */
export const ImagePanel = ({ index, title, src, filename, badge }) => (
  <figure className="flex min-w-0 flex-col gap-2">
    <div className="flex items-center justify-between gap-2 text-sm font-medium text-slate-800">
      <span>{index != null && <span className="text-slate-400">{index}. </span>}{title}</span>
      {src && filename && (
        <button onClick={() => download(src, filename)} className="font-mono text-[11px] font-semibold uppercase text-brand-600 hover:text-brand-700">↓ PNG</button>
      )}
    </div>
    <div className="relative aspect-square w-full overflow-hidden rounded-xl border border-slate-200 bg-slate-100">
      {src ? (
        <img src={src} alt={title} className="h-full w-full object-contain" />
      ) : (
        <div className="flex h-full items-center justify-center font-mono text-xs text-slate-400">no image yet</div>
      )}
      {badge && <span className="absolute left-2 top-2 rounded-md bg-slate-900/70 px-2 py-0.5 font-mono text-[10px] text-white">{badge}</span>}
    </div>
  </figure>
);

/** Horizontal probability/weight rows. `selected` (key) gets the highlighted "Selected" treatment. */
export const WeightBars = ({ data, selected, selectedLabel = "Selected" }) => {
  const entries = Object.entries(data);
  const max = Math.max(...entries.map(([, v]) => v));
  return (
    <ul className="space-y-2">
      {entries.map(([k, v]) => {
        const top = selected ? k === selected : v === max;
        return (
          <li key={k} className={`rounded-xl px-3 py-2.5 ${top ? "bg-brand-50/70" : ""}`}>
            <div className="mb-1.5 flex items-center justify-between gap-2 text-sm">
              <span className="flex flex-wrap items-center gap-2">
                <span className={top ? "font-semibold text-slate-900" : "text-slate-600"}>{k}</span>
                {top && <span className="rounded-full bg-brand-600 px-2 py-0.5 font-mono text-[10px] font-medium text-white">{selectedLabel}</span>}
              </span>
              <Mono className={top ? "font-semibold text-brand-700" : "text-slate-600"}>{(v * 100).toFixed(1)}%</Mono>
            </div>
            <div className="h-2 overflow-hidden rounded-full bg-slate-100">
              <div className={`h-full rounded-full transition-all duration-500 ${top ? "bg-brand-600" : "bg-slate-400"}`} style={{ width: `${Math.max(v * 100, 0.6)}%` }} />
            </div>
          </li>
        );
      })}
    </ul>
  );
};

export const ErrorBox = ({ msg }) =>
  msg ? <div role="alert" className="rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">{msg}</div> : null;
