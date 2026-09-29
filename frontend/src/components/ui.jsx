import React from "react";
import { download } from "../api.js";

export const Card = ({ title, children, className = "", right }) => (
  <section className={`rounded-2xl border border-slate-200 bg-white p-5 shadow-sm ${className}`}>
    {(title || right) && (
      <div className="mb-4 flex items-center justify-between gap-2">
        <h3 className="text-sm font-semibold uppercase tracking-wide text-slate-500">{title}</h3>
        {right}
      </div>
    )}
    {children}
  </section>
);

export const Button = ({ variant = "primary", className = "", ...p }) => (
  <button
    {...p}
    className={`inline-flex items-center justify-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition disabled:cursor-not-allowed disabled:opacity-50 ${
      variant === "primary"
        ? "bg-brand-600 text-white hover:bg-brand-700"
        : "border border-slate-300 bg-white text-slate-700 hover:bg-slate-100"
    } ${className}`}
  />
);

export const Stat = ({ label, value, hint }) => (
  <div className="rounded-xl bg-slate-50 px-4 py-3">
    <div className="text-xs uppercase tracking-wide text-slate-500">{label}</div>
    <div className="mt-0.5 text-lg font-semibold text-slate-900">{value}</div>
    {hint && <div className="text-xs text-slate-500">{hint}</div>}
  </div>
);

/** One image with caption and optional download button. Images are 128px, shown enlarged. */
export const ImagePanel = ({ title, src, filename, badge }) => (
  <figure className="flex flex-col items-center gap-2">
    <div className="relative aspect-square w-full max-w-[280px] overflow-hidden rounded-xl border border-slate-200 bg-slate-100">
      {src ? (
        <img src={src} alt={title} className="h-full w-full object-contain [image-rendering:auto]" />
      ) : (
        <div className="flex h-full items-center justify-center text-xs text-slate-400">no image yet</div>
      )}
      {badge && <span className="absolute left-2 top-2 rounded-full bg-black/60 px-2 py-0.5 text-[11px] text-white">{badge}</span>}
    </div>
    <figcaption className="flex items-center gap-2 text-sm text-slate-600">
      {title}
      {src && filename && (
        <button onClick={() => download(src, filename)} className="text-brand-600 underline underline-offset-2 hover:text-brand-700">
          download
        </button>
      )}
    </figcaption>
  </figure>
);

/** Horizontal probability/weight bars; the largest entry is highlighted. */
export const WeightBars = ({ data, highlight }) => {
  const entries = Object.entries(data);
  const max = Math.max(...entries.map(([, v]) => v));
  return (
    <ul className="space-y-2">
      {entries.map(([k, v]) => {
        const top = highlight ? k === highlight : v === max;
        return (
          <li key={k}>
            <div className="mb-1 flex justify-between text-sm">
              <span className={top ? "font-semibold text-slate-900" : "text-slate-600"}>{k}</span>
              <span className="tabular-nums text-slate-600">{(v * 100).toFixed(1)}%</span>
            </div>
            <div className="h-2.5 overflow-hidden rounded-full bg-slate-100">
              <div className={`h-full rounded-full transition-all ${top ? "bg-brand-600" : "bg-slate-300"}`} style={{ width: `${Math.max(v * 100, 0.5)}%` }} />
            </div>
          </li>
        );
      })}
    </ul>
  );
};

export const ErrorBox = ({ msg }) =>
  msg ? <div role="alert" className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">{msg}</div> : null;
