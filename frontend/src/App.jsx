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
    <div className="flex min-h-screen flex-col lg:h-screen">
      <header className="flex h-14 shrink-0 items-center justify-between gap-3 border-b border-slate-200 bg-white px-4">
        <div className="flex min-w-0 items-center gap-3">
          <div className="grid h-8 w-8 shrink-0 place-items-center rounded-lg bg-brand-600 text-white">✦</div>
          <div className="leading-tight">
            <div className="text-[17px] font-semibold text-slate-900">Restoration Studio</div>
            <div className="hidden font-mono text-[10px] uppercase tracking-widest text-slate-500 sm:block">Vision ML Lab</div>
          </div>
          <span className="ml-2 hidden items-center gap-2 rounded-full bg-slate-50 px-3 py-1 font-mono text-xs text-slate-600 md:flex">
            <span className={`h-2 w-2 rounded-full ${down ? "bg-rose-500" : "bg-emerald-500"}`} />
            {current.task} · {current.label}
            <span className="text-slate-300">|</span>
            {health ? <>ONNX Runtime {health.onnxruntime} · <span className="text-cyan-diag">{device}</span></> : "connecting…"}
          </span>
        </div>
        <a href="/api/docs" target="_blank" rel="noreferrer" className="rounded-lg bg-slate-100 px-3 py-1.5 font-mono text-xs font-medium text-slate-700 hover:bg-slate-200">{"</>"} API / Docs</a>
      </header>

      <div className="flex min-h-0 flex-1 flex-col lg:flex-row">
        <aside className="shrink-0 border-b border-slate-200 bg-white lg:flex lg:w-64 lg:flex-col lg:border-b-0 lg:border-r">
          <div className="hidden px-5 pb-1 pt-4 font-mono text-[10px] font-semibold uppercase tracking-widest text-slate-400 lg:block">Workspaces</div>
          <nav className="flex gap-1 overflow-x-auto p-2 lg:flex-col lg:overflow-visible">
            {WORKSPACES.map((w) => (
              <button key={w.id} onClick={() => setActive(w.id)}
                className={`flex shrink-0 items-center justify-between gap-2 rounded-xl px-3 py-2 text-left transition ${active === w.id ? "bg-brand-50 ring-1 ring-brand-100" : "hover:bg-slate-50"}`}>
                <span>
                  <span className={`block text-[14px] font-medium leading-tight ${active === w.id ? "text-brand-700" : "text-slate-800"}`}>{w.label}</span>
                  <span className="hidden text-xs text-slate-500 lg:block">{w.sub}</span>
                </span>
                {health && (ready(w.id) ? <span title="models loaded" className="h-2 w-2 shrink-0 rounded-full bg-emerald-500" /> : <Pill tone="warn">Missing</Pill>)}
              </button>
            ))}
          </nav>
          <div className="mt-auto hidden p-3 lg:block">
            <div className="rounded-xl bg-slate-50 p-3 font-mono text-xs text-slate-600">
              <div className="mb-2 flex justify-between text-[10px] font-semibold uppercase tracking-widest text-slate-500">
                <span>Engine core</span>
                {down ? <span className="text-rose-600">● offline</span> : <span className="text-emerald-600">● live</span>}
              </div>
              {health ? (
                <dl className="space-y-1">
                  <div className="flex justify-between"><dt>Runtime</dt><dd className="text-slate-900">ORT {health.onnxruntime}</dd></div>
                  <div className="flex justify-between"><dt>Device</dt><dd className="text-slate-900">{device}</dd></div>
                  <div className="flex justify-between"><dt>Models</dt><dd className="text-brand-600">{loaded}/{models.length} loaded</dd></div>
                </dl>
              ) : <span>{down ? "backend unreachable" : "connecting…"}</span>}
            </div>
          </div>
        </aside>

        <main className="flex min-h-0 min-w-0 flex-1 flex-col p-3 sm:p-4">
          {down && <div className="mb-3 shrink-0 rounded-lg border border-rose-200 bg-rose-50 px-4 py-2 text-sm text-rose-700">Cannot reach the backend API. Is it running?</div>}
          {health && !ready(current.id) && (
            <div className="mb-3 shrink-0 rounded-lg border border-amber-200 bg-amber-50 px-4 py-2 text-sm text-amber-800">
              Some ONNX models are missing: <Mono>{models.filter((m) => !m.loaded).map((m) => m.file).join(", ")}</Mono>
            </div>
          )}
          <div key={current.id} className="min-h-0 flex-1">{current.el}</div>
        </main>
      </div>
    </div>
  );
}
