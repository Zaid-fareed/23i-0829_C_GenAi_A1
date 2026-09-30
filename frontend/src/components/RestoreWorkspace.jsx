import React, { useState } from "react";
import { post } from "../api.js";
import ImageSource from "./ImageSource.jsx";
import CorruptionControls, { DEFAULT_CORRUPTION } from "./CorruptionControls.jsx";
import { Button, Card, ErrorBox, ImagePanel, Mono, Pill, Segmented, Stat, WeightBars } from "./ui.jsx";

const KINDS = {
  universal: {
    title: "Universal Restoration",
    blurb: "One denoising autoencoder restores clean, salt-and-pepper, blurred and occluded images without being told the corruption type.",
    path: "/api/restore/universal",
  },
  hard: {
    title: "Hard-Routed Restoration",
    blurb: "A classifier predicts the corruption type and sends the image to exactly one specialist autoencoder. Clean images bypass restoration.",
    path: "/api/restore/hard",
  },
  soft: {
    title: "Soft Mixture-of-Experts Restoration",
    blurb: "A gating network gives every branch (identity + 3 experts) a continuous weight; the output is their weighted sum.",
    path: "/api/restore/soft",
  },
};

const settingsText = (c) => {
  if (!c?.applied) return "none (image used as uploaded)";
  const parts = [c.type, c.severity];
  if (c.p != null) parts.push(`p=${c.p}`);
  if (c.kernel != null) parts.push(`kernel ${c.kernel}, sigma ${c.sigma}`);
  if (c.coverage != null) parts.push(`${c.n_rects} rect(s), ${(c.coverage * 100).toFixed(1)}% covered`);
  return parts.join(" · ");
};

const STRIP = ["bg-slate-500", "bg-rose-500", "bg-sky-500", "bg-emerald-500"];

export default function RestoreWorkspace({ kind }) {
  const cfg = KINDS[kind];
  const [src, setSrc] = useState({ file: null, sample: null, previewUrl: null, name: "" });
  const [cor, setCor] = useState(DEFAULT_CORRUPTION);
  const [mode, setMode] = useState("predicted");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [res, setRes] = useState(null);
  const [copied, setCopied] = useState(false);

  const run = async () => {
    setBusy(true); setErr("");
    try {
      const fields = { ...cor, ...(kind === "hard" ? { mode } : {}) };
      setRes(await post(cfg.path, { file: src.file, sample: src.sample, fields }));
    } catch (e) { setErr(e.message); setRes(null); }
    finally { setBusy(false); }
  };
  const ready = src.file || src.sample;
  const ms = res && (kind === "hard" ? res.timing_ms.total : res.inference_ms);
  const topProb = res && kind === "hard" ? Math.max(...Object.values(res.probabilities)) : null;

  const copyJson = async () => {
    const { input_image, output_image, clean_image, quality, ...rest } = res;
    const payload = { ...rest, quality: quality && { psnr_corrupted_db: quality.psnr_corrupted_db, psnr_restored_db: quality.psnr_restored_db } };
    try { await navigator.clipboard.writeText(JSON.stringify(payload, null, 2)); setCopied(true); setTimeout(() => setCopied(false), 1500); } catch { /* clipboard blocked */ }
  };

  return (
    <div className="space-y-6">
      <header>
        <h2 className="text-[28px] font-bold leading-9 tracking-tight text-slate-900">{cfg.title}</h2>
        <p className="mt-1 max-w-3xl text-slate-600">{cfg.blurb}</p>
      </header>

      <div className="grid gap-6 xl:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
        <div className="space-y-6">
          <Card step={1} title="Input image" tag="Source">
            <ImageSource value={src} onChange={(s) => { setSrc(s); setRes(null); }} />
          </Card>
          <Card step={2} title="Runtime corruption" tag="Synthesizer">
            <CorruptionControls value={cor} onChange={setCor} />
            <p className="mt-3 text-xs text-slate-400">Choose “None” to restore an image that is already corrupted.</p>
          </Card>
          {kind === "hard" && (
            <Card step={3} title="Routing mode">
              <Segmented options={[["predicted", "Predicted (classifier)"], ["oracle", "Oracle (true label)"]]} value={mode} onChange={setMode} />
              <p className="mt-2 text-xs text-slate-400">{mode === "predicted" ? "The classifier chooses the expert." : "The applied corruption label chooses the expert (needs a corruption type)."}</p>
            </Card>
          )}
          <Button className="w-full !py-3.5 text-base" onClick={run} disabled={!ready || busy}>
            {busy ? "Running…" : "▷  Run model"}
            {res && !busy && <Mono className="rounded-full bg-white/20 px-2 py-0.5 text-xs">~{ms} ms</Mono>}
          </Button>
          <ErrorBox msg={err} />
        </div>

        <div className="space-y-6">
          <Card title="Restoration result" right={res ? <Pill tone="ok">Done · {ms} ms</Pill> : <Pill tone="warn">Waiting for run</Pill>}>
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
              <ImagePanel index={1} title="Model input" src={res?.input_image} filename="input.png" badge={res && (res.corruption.applied ? "corrupted" : "as uploaded")} />
              <ImagePanel index={2} title="Restored output" src={res?.output_image} filename="restored.png" badge={res && "output"} />
              {res?.quality
                ? <ImagePanel index={3} title="Absolute error map" src={res.quality.error_map} filename="error_map.png" badge="|restored − clean|" />
                : <ImagePanel index={3} title="Absolute error map" src={null} />}
            </div>
            {res?.clean_image && (
              <details className="mt-4 text-sm text-slate-600">
                <summary className="cursor-pointer">Show original (clean) image</summary>
                <img src={res.clean_image} alt="clean" className="mt-2 h-32 w-32 rounded-lg border" />
              </details>
            )}
          </Card>

          {res && kind === "hard" && (
            <Card title="Classifier probabilities" right={<Mono className="text-xs text-slate-500">mode: {res.mode}</Mono>}>
              <p className="mb-3 text-sm text-slate-500">Softmax distribution over the four input classes. Decision policy: <Mono className="rounded bg-brand-50 px-1.5 py-0.5 text-brand-700">argmax</Mono></p>
              <WeightBars data={res.probabilities} selected={res.predicted} selectedLabel="Predicted" />
              {res.misrouted && <p className="mt-3 rounded-md bg-amber-50 px-3 py-2 text-sm text-amber-800">Classifier error: predicted “{res.predicted}” but the applied corruption was “{res.corruption.type}”.</p>}
            </Card>
          )}

          {res && kind === "soft" && (
            <Card title="Routing weights (gate output)">
              <WeightBars data={res.weights} selected={res.dominant} selectedLabel="Dominant" />
              <div className="mt-4">
                <div className="mb-1 font-mono text-[10px] font-semibold uppercase tracking-widest text-slate-500">Contribution to the final image</div>
                <div className="flex h-6 overflow-hidden rounded-md text-[10px] text-white">
                  {Object.entries(res.weights).map(([k, v], i) => (
                    <div key={k} title={`${k}: ${(v * 100).toFixed(1)}%`} style={{ width: `${v * 100}%` }} className={STRIP[i]}>
                      {v > 0.12 && <span className="px-1 font-mono">{k.split(" ")[0]}</span>}
                    </div>
                  ))}
                </div>
              </div>
            </Card>
          )}

          {res && (
            <Card title="System information" right={<span className="font-mono text-[10px] uppercase tracking-widest text-slate-500">ONNX Runtime · CPU</span>}>
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                <Stat label="Inference time" value={`${ms} ms`} hint={kind === "hard" ? `classifier ${res.timing_ms.classifier} ms · expert ${res.timing_ms.expert} ms` : "single forward pass"} />
                {res.quality
                  ? <Stat label="PSNR before → after" value={`${res.quality.psnr_corrupted_db} → ${res.quality.psnr_restored_db} dB`} hint={`${(res.quality.psnr_restored_db - res.quality.psnr_corrupted_db) >= 0 ? "+" : ""}${(res.quality.psnr_restored_db - res.quality.psnr_corrupted_db).toFixed(1)} dB change`} />
                  : <Stat label="PSNR before → after" value="n/a" hint="needs a clean reference (apply a corruption)" />}
                {kind === "hard" && <Stat label="Selected expert" value={res.expert} hint={`predicted class: ${res.predicted}`} />}
                {kind === "hard" && <Stat label="Routing confidence" value={topProb.toFixed(3)} hint={topProb > 0.9 ? "high confidence" : "low confidence"} />}
                {kind === "soft" && <Stat label="Dominant branch" value={res.dominant} hint={`${(res.dominant_weight * 100).toFixed(1)}% of the mixture`} />}
                {kind === "soft" && <Stat label="Weight entropy" value={`${res.entropy} / ${res.max_entropy}`} hint={res.entropy > 0.6 * res.max_entropy ? "distributed across experts" : "one expert dominates"} />}
              </div>
              <div className="mt-3 flex flex-wrap items-center justify-between gap-2 rounded-lg bg-slate-50 px-3 py-2 text-xs text-slate-600">
                <span className="font-mono">PARAMS: {settingsText(res.corruption)}</span>
                <button onClick={copyJson} className="font-mono font-semibold text-brand-600 hover:text-brand-700">{copied ? "Copied ✓" : "Copy JSON"}</button>
              </div>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
