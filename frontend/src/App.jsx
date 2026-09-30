import React, { useEffect, useState } from "react";
import { getHealth } from "./api.js";
import RestoreWorkspace from "./components/RestoreWorkspace.jsx";
import SketchWorkspace from "./components/SketchWorkspace.jsx";
import { Mono, Pill } from "./components/ui.jsx";

const WORKSPACES = [
  { id: "universal", label: "Universal Restoration", sub: "Single-model baseline", task: "Task 1", el: <RestoreWorkspace kind="universal" /> },
  { id: "hard", label: "Hard-Routed Restoration", sub: "Classifier gated routing", task: "Task 2", el: <RestoreWorkspace kind="hard" /> },
  { id: "soft", label: "Soft Mixture-of-Experts", sub: "Dynamic expert blending", task: "Task 3", el: <RestoreWorkspace kind="soft" /> },
  { id: "sketch", label: "Face-to-Sketch Generator", sub: "cGAN paired synthesis", task: "Task 4", el: <SketchWorkspace /> },
];

export default function App() {
  const [active, setActive] = useState(() => { try { return localStorage.getItem("ws") || "universal"; } catch { return "universal"; } });
  const [health, setHealth] = useState(null);
  const [down, setDown] = useState(false);

  useEffect(() => {
    const poll = () => getHealth().then((h) => { setHealth(h); setDown(false); }).catch(() => setDown(true));
    poll();
    const t = setInterval(poll, 15000);
    return () => clearInterval(t);
  }, []);
  useEffect(() => { try { localStorage.setItem("ws", active); } catch { /* ignore */ } }, [active]);

  const current = WORKSPACES.find((w) => w.id === active) || WORKSPACES[0];
  const ready = (id) => health?.workspaces?.[id];
  const models = health ? Object.values(health.models) : [];
  const loaded = models.filter((m) => m.loaded).length;
  const device = health?.providers?.[0]?.replace("ExecutionProvider", "").toUpperCase();

  return (
    <div className="min-h-screen lg:flex">
      <aside className="border-b border-slate-200 bg-white lg:sticky lg:top-0 lg:flex lg:h-screen lg:w-80 lg:shrink-0 lg:flex-col lg:border-b-0 lg:border-r">
        <div className="flex items-center gap-3 px-5 pb-3 pt-5">
          <div className="grid h-10 w-10 place-items-center rounded-xl bg-brand-600 text-lg text-white">✦</div>
          <div>
            <div className="text-lg font-semibold leading-tight text-slate-900">Restoration Studio</div>
            <div className="font-mono text-[10px] uppercase tracking-widest text-slate-500">Vision ML Lab</div>
          </div>
        </div>
        <div className="hidden px-5 pb-2 pt-3 font-mono text-[10px] font-semibold uppercase tracking-widest text-slate-400 lg:block">Workspaces</div>
        <nav className="flex gap-1 overflow-x-auto px-3 pb-3 lg:flex-col lg:overflow-visible">
          {WORKSPACES.map((w) => (
            <button key={w.id} onClick={() => setActive(w.id)}
              className={`flex shrink-0 items-center justify-between gap-3 rounded-xl px-3 py-2.5 text-left transition ${active === w.id ? "bg-brand-50 ring-1 ring-brand-100" : "hover:bg-slate-50"}`}>
              <span>
                <span className={`block text-[15px] font-medium ${active === w.id ? "text-brand-700" : "text-slate-800"}`}>{w.label}</span>
                <span className="hidden text-sm text-slate-500 lg:block">{w.sub}</span>
              </span>
              {health && (ready(w.id) ? <Pill tone="ok" pulse>Ready</Pill> : <Pill tone="warn">Missing</Pill>)}
            </button>
          ))}
        </nav>
        <div className="mt-auto hidden p-4 lg:block">
          <div className="rounded-xl bg-slate-50 p-3 font-mono text-xs text-slate-600">
            <div className="mb-2 flex justify-between text-[10px] font-semibold uppercase tracking-widest text-slate-500">
              <span>Engine core</span>
              {down ? <span className="text-rose-600">● offline</span> : <span className="text-emerald-600">● live</span>}
            </div>
            {health ? (
              <dl className="space-y-1">
                <div className="flex justify-between"><dt>Runtime</dt><dd className="text-slate-900">ONNX Runtime {health.onnxruntime}</dd></div>
                <div className="flex justify-between"><dt>Device</dt><dd className="text-slate-900">{device}</dd></div>
                <div className="flex justify-between"><dt>Models</dt><dd className="text-brand-600">{loaded}/{models.length} loaded</dd></div>
              </dl>
            ) : <span>{down ? "backend unreachable" : "connecting…"}</span>}
          </div>
        </div>
      </aside>

      <div className="min-w-0 flex-1">
        <header className="flex items-center justify-between gap-3 border-b border-slate-200 bg-white/80 px-4 py-3 backdrop-blur sm:px-8">
          <nav className="truncate font-mono text-sm text-slate-500">
            restoration-studio <span className="text-slate-300">›</span> <span className="text-brand-600">{current.task.toLowerCase().replace(" ", "-")}</span> <span className="text-slate-300">›</span> <span className="text-slate-900">{current.id}</span>
          </nav>
          <a href="/api/docs" target="_blank" rel="noreferrer" className="rounded-lg bg-slate-100 px-3 py-1.5 font-mono text-xs font-medium text-slate-700 hover:bg-slate-200">{"</>"} API / Docs</a>
        </header>

        <main className="p-4 sm:p-8">
          <div className="mb-6 flex flex-wrap items-center justify-between gap-2 rounded-xl border border-slate-200 bg-white px-4 py-2.5 font-mono text-xs text-slate-600">
            <span className="flex items-center gap-2"><span className={`h-2 w-2 rounded-full ${down ? "bg-rose-500" : "bg-emerald-500"}`} />{current.task} <span className="text-slate-300">/</span> {current.label}</span>
            <span>Backend: <span className="text-slate-900">{health ? `ONNX Runtime ${health.onnxruntime}` : "…"}</span> <span className="px-2 text-slate-300">|</span> Device: <span className="text-cyan-diag">{device || "…"}</span></span>
          </div>
          {down && <div className="mb-4 rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">Cannot reach the backend API. Is it running?</div>}
          {health && !ready(current.id) && (
            <div className="mb-4 rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800">
              Some ONNX models are missing: <Mono>{models.filter((m) => !m.loaded).map((m) => m.file).join(", ")}</Mono>
            </div>
          )}
          <div key={current.id}>{current.el}</div>
        </main>
      </div>
    </div>
  );
}
