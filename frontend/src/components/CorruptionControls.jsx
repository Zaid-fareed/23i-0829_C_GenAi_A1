import React from "react";
import { Mono, Segmented } from "./ui.jsx";

const TYPES = [
  ["none", "None (use image as-is)"],
  ["salt_pepper", "Salt-and-pepper (impulse noise)"],
  ["blur", "Gaussian blur"],
  ["occlusion", "Rectangular occlusion"],
];
const SEVS = [["low", "Low"], ["medium", "Medium"], ["high", "High"], ["custom", "Custom"]];
const FIXED_TEXT = {
  salt_pepper: { low: "p = 0.03", medium: "p = 0.08", high: "p = 0.15" },
  blur: { low: "kernel 3 · σ 0.7", medium: "kernel 5 · σ 1.5", high: "kernel 7 · σ 2.5" },
  occlusion: { low: "1 rect · ~10%", medium: "2 rects · ~20%", high: "3 rects · ~35%" },
};

const Label = ({ children, right }) => (
  <div className="mb-1 flex items-center justify-between font-mono text-[10px] font-semibold uppercase tracking-widest text-slate-500">
    <span>{children}</span>{right && <span className="normal-case tracking-normal text-brand-600">{right}</span>}
  </div>
);

const Slider = ({ label, min, max, step, value, onChange, fmt = (v) => v }) => (
  <label className="block text-sm">
    <span className="flex justify-between text-slate-700"><span>{label}</span><Mono className="text-slate-900">{fmt(value)}</Mono></span>
    <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} className="w-full accent-brand-600" />
  </label>
);

/** Runtime corruption controls (applied server-side to the input). `value` is a plain object. */
export default function CorruptionControls({ value, onChange }) {
  const set = (k, v) => onChange({ ...value, [k]: v });
  const c = value.corruption;
  return (
    <div className="space-y-4">
      <div>
        <Label>Corruption type</Label>
        <select value={c} onChange={(e) => set("corruption", e.target.value)}
          className="w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-2.5 text-sm text-slate-900 focus:border-brand-500 focus:outline-none focus:ring-[3px] focus:ring-brand-100">
          {TYPES.map(([k, l]) => <option key={k} value={k}>{l}</option>)}
        </select>
      </div>
      {c !== "none" && (
        <>
          <div>
            <Label right={FIXED_TEXT[c]?.[value.severity]}>Degradation severity</Label>
            <Segmented options={SEVS} value={value.severity} onChange={(v) => set("severity", v)} />
          </div>
          {value.severity === "custom" && (
            <div className="space-y-3 rounded-lg bg-slate-50 p-3">
              {c === "salt_pepper" && <Slider label="Corrupted pixel probability" min={0.01} max={0.3} step={0.01} value={value.sp_p} onChange={(v) => set("sp_p", v)} fmt={(v) => v.toFixed(2)} />}
              {c === "blur" && (<>
                <Slider label="Kernel size" min={3} max={15} step={2} value={value.blur_kernel} onChange={(v) => set("blur_kernel", v)} />
                <Slider label="Sigma" min={0.3} max={4} step={0.1} value={value.blur_sigma} onChange={(v) => set("blur_sigma", v)} fmt={(v) => v.toFixed(1)} />
              </>)}
              {c === "occlusion" && (<>
                <Slider label="Area covered" min={0.05} max={0.5} step={0.01} value={value.occ_coverage} onChange={(v) => set("occ_coverage", v)} fmt={(v) => `${Math.round(v * 100)}%`} />
                <Slider label="Rectangles" min={1} max={5} step={1} value={value.occ_rects} onChange={(v) => set("occ_rects", v)} />
              </>)}
            </div>
          )}
          <div>
            <Label>Random seed</Label>
            <div className="flex gap-2">
              <input type="number" value={value.seed} onChange={(e) => set("seed", Number(e.target.value) || 0)}
                className="w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-2.5 font-mono text-sm text-slate-900 focus:border-brand-500 focus:outline-none focus:ring-[3px] focus:ring-brand-100" />
              <button type="button" title="Randomize seed" onClick={() => set("seed", Math.floor(Math.random() * 100000))}
                className="rounded-lg border border-slate-200 bg-white px-3 text-slate-600 hover:bg-slate-100">⚄</button>
            </div>
            <p className="mt-1 text-xs text-slate-400">Same seed = same corruption.</p>
          </div>
        </>
      )}
    </div>
  );
}

export const DEFAULT_CORRUPTION = {
  corruption: "salt_pepper", severity: "medium", seed: 42, sp_p: 0.08, blur_kernel: 5, blur_sigma: 1.5, occ_coverage: 0.2, occ_rects: 2,
};
