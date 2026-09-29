import React from "react";

const TYPES = [
  ["none", "None (use image as-is)"],
  ["salt_pepper", "Salt-and-pepper noise"],
  ["blur", "Gaussian blur"],
  ["occlusion", "Rectangular occlusion"],
];
const SEVS = [["low", "Low"], ["medium", "Medium"], ["high", "High"], ["custom", "Custom"]];

const Slider = ({ label, min, max, step, value, onChange, fmt = (v) => v }) => (
  <label className="block text-sm">
    <span className="flex justify-between text-slate-600"><span>{label}</span><span className="tabular-nums">{fmt(value)}</span></span>
    <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} className="w-full accent-brand-600" />
  </label>
);

/** Controls the runtime corruption applied server-side to the (clean) input. `value` is a plain object. */
export default function CorruptionControls({ value, onChange }) {
  const set = (k, v) => onChange({ ...value, [k]: v });
  const c = value.corruption;
  return (
    <div className="space-y-3">
      <label className="block text-sm">
        <span className="text-slate-600">Corruption</span>
        <select value={c} onChange={(e) => set("corruption", e.target.value)} className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-3 py-2">
          {TYPES.map(([k, l]) => <option key={k} value={k}>{l}</option>)}
        </select>
      </label>
      {c !== "none" && (
        <>
          <div>
            <span className="text-sm text-slate-600">Severity</span>
            <div className="mt-1 grid grid-cols-4 gap-1">
              {SEVS.map(([k, l]) => (
                <button key={k} onClick={() => set("severity", k)}
                  className={`rounded-md border px-2 py-1.5 text-sm ${value.severity === k ? "border-brand-600 bg-brand-50 font-medium text-brand-700" : "border-slate-300 text-slate-600"}`}>{l}</button>
              ))}
            </div>
          </div>
          {value.severity === "custom" && (
            <div className="space-y-2 rounded-lg bg-slate-50 p-3">
              {c === "salt_pepper" && <Slider label="Pixel probability" min={0.01} max={0.3} step={0.01} value={value.sp_p} onChange={(v) => set("sp_p", v)} fmt={(v) => v.toFixed(2)} />}
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
          <label className="block text-sm">
            <span className="text-slate-600">Random seed (same seed = same corruption)</span>
            <input type="number" value={value.seed} onChange={(e) => set("seed", Number(e.target.value) || 0)} className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2" />
          </label>
        </>
      )}
    </div>
  );
}

export const DEFAULT_CORRUPTION = {
  corruption: "salt_pepper", severity: "medium", seed: 42, sp_p: 0.08, blur_kernel: 5, blur_sigma: 1.5, occ_coverage: 0.2, occ_rects: 2,
};
