import React, { useEffect, useState } from "react";
import { getHealth } from "./api.js";
import RestoreWorkspace from "./components/RestoreWorkspace.jsx";
import SketchWorkspace from "./components/SketchWorkspace.jsx";

const WORKSPACES = [
  { id: "universal", label: "Universal Restoration", task: "Task 1", el: <RestoreWorkspace kind="universal" /> },
  { id: "hard", label: "Hard-Routed Restoration", task: "Task 2", el: <RestoreWorkspace kind="hard" /> },
  { id: "soft", label: "Soft Mixture-of-Experts", task: "Task 3", el: <RestoreWorkspace kind="soft" /> },
  { id: "sketch", label: "Face-to-Sketch Generator", task: "Task 4", el: <SketchWorkspace /> },
];

export default function App() {
  const [active, setActive] = useState(() => localStorage.getItem("ws") || "universal");
  const [health, setHealth] = useState(null);
  const [down, setDown] = useState(false);

  useEffect(() => {
    const poll = () => getHealth().then((h) => { setHealth(h); setDown(false); }).catch(() => setDown(true));
    poll();
    const t = setInterval(poll, 15000);
    return () => clearInterval(t);
  }, []);
  useEffect(() => { try { localStorage.setItem("ws", active); } catch { /* ignore */ } }, [active]);

  const current = WORKSPACES.find((w) => w.id === active);
  const ready = (id) => health?.workspaces?.[id];

  return (
    <div className="min-h-screen lg:flex">
      <aside className="border-b border-slate-200 bg-white lg:sticky lg:top-0 lg:h-screen lg:w-72 lg:shrink-0 lg:border-b-0 lg:border-r">
        <div className="px-5 pb-3 pt-5">
          <div className="text-lg font-bold text-brand-700">Restoration Studio</div>
          <div className="text-xs text-slate-500">Generative AI · Assignment 1</div>
        </div>
        <nav className="flex gap-1 overflow-x-auto px-3 pb-3 lg:flex-col lg:overflow-visible">
          {WORKSPACES.map((w) => (
            <button key={w.id} onClick={() => setActive(w.id)}
              className={`flex shrink-0 items-center justify-between gap-3 rounded-lg px-3 py-2.5 text-left text-sm ${active === w.id ? "bg-brand-50 font-medium text-brand-700" : "text-slate-600 hover:bg-slate-100"}`}>
              <span><span className="block text-[11px] uppercase tracking-wide text-slate-400">{w.task}</span>{w.label}</span>
              {health && <span title={ready(w.id) ? "models loaded" : "models missing"} className={`h-2 w-2 shrink-0 rounded-full ${ready(w.id) ? "bg-emerald-500" : "bg-amber-500"}`} />}
            </button>
          ))}
        </nav>
        <div className="hidden px-5 py-4 text-xs text-slate-500 lg:block">
          <div className="mb-1 font-semibold uppercase tracking-wide">Backend</div>
          {down ? <span className="text-red-600">unreachable</span> : health ? (
            <ul className="space-y-0.5">
              <li>status: {health.status}</li>
              <li>runtime: ONNX Runtime {health.onnxruntime}</li>
              <li>device: {health.providers?.[0]?.replace("ExecutionProvider", "")}</li>
              <li>models: {Object.values(health.models).filter((m) => m.loaded).length}/{Object.keys(health.models).length} loaded</li>
            </ul>
          ) : "connecting…"}
        </div>
      </aside>

      <main className="min-w-0 flex-1 p-4 sm:p-8">
        {down && <div className="mb-4 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">Cannot reach the backend API. Is it running?</div>}
        {health && !ready(active) && (
          <div className="mb-4 rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800">
            Some ONNX models for this workspace are missing: {Object.values(health.models).filter((m) => !m.loaded).map((m) => m.file).join(", ")}
          </div>
        )}
        <div key={current.id}>{current.el}</div>
      </main>
    </div>
  );
}
