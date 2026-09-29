import React, { useState } from "react";
import { post } from "../api.js";
import ImageSource from "./ImageSource.jsx";
import CorruptionControls, { DEFAULT_CORRUPTION } from "./CorruptionControls.jsx";
import { Button, Card, ErrorBox, ImagePanel, Stat, WeightBars } from "./ui.jsx";

const KINDS = {
  universal: {
    title: "Universal Restoration",
    blurb: "One denoising autoencoder restores clean, salt-and-pepper, blurred and occluded images without being told the corruption type.",
    path: "/api/restore/universal",
  },
  hard: {
    title: "Hard-Routed Restoration",
    blurb: "A classifier predicts the corruption type and sends the image to exactly one specialist autoencoder (clean images bypass restoration).",
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

export default function RestoreWorkspace({ kind }) {
  const cfg = KINDS[kind];
  const [src, setSrc] = useState({ file: null, sample: null, previewUrl: null });
  const [cor, setCor] = useState(DEFAULT_CORRUPTION);
  const [mode, setMode] = useState("predicted");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [res, setRes] = useState(null);

  const run = async () => {
    setBusy(true); setErr("");
    try {
      const fields = { ...cor, ...(kind === "hard" ? { mode } : {}) };
      setRes(await post(cfg.path, { file: src.file, sample: src.sample, fields }));
    } catch (e) { setErr(e.message); setRes(null); }
    finally { setBusy(false); }
  };
  const ready = src.file || src.sample;
  const timing = res && (kind === "hard" ? `${res.timing_ms.total} ms` : `${res.inference_ms} ms`);

  return (
    <div className="space-y-6">
      <header>
        <h2 className="text-2xl font-semibold text-slate-900">{cfg.title}</h2>
        <p className="mt-1 max-w-3xl text-slate-600">{cfg.blurb}</p>
      </header>

      <div className="grid gap-6 lg:grid-cols-[360px_1fr]">
        <div className="space-y-6">
          <Card title="1 · Input image">
            <ImageSource value={src} onChange={(s) => { setSrc(s); setRes(null); }} />
            {src.previewUrl && <img src={src.previewUrl} alt="selected" className="mt-4 h-28 w-28 rounded-lg border object-cover" />}
          </Card>
          <Card title="2 · Runtime corruption">
            <CorruptionControls value={cor} onChange={setCor} />
            <p className="mt-3 text-xs text-slate-500">Choose “None” to restore an image that is already corrupted.</p>
          </Card>
          {kind === "hard" && (
            <Card title="3 · Routing mode">
              <div className="grid grid-cols-2 gap-1">
                {[["predicted", "Predicted (classifier)"], ["oracle", "Oracle (true label)"]].map(([k, l]) => (
                  <button key={k} onClick={() => setMode(k)} className={`rounded-md border px-2 py-1.5 text-sm ${mode === k ? "border-brand-600 bg-brand-50 font-medium text-brand-700" : "border-slate-300 text-slate-600"}`}>{l}</button>
                ))}
              </div>
            </Card>
          )}
          <Button className="w-full" onClick={run} disabled={!ready || busy}>{busy ? "Running…" : "Run model"}</Button>
          <ErrorBox msg={err} />
        </div>

        <div className="space-y-6">
          <Card title="Result">
            <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
              <ImagePanel title="Model input" src={res?.input_image} filename="input.png" badge={res?.corruption?.applied ? "corrupted" : "as uploaded"} />
              <ImagePanel title="Restored output" src={res?.output_image} filename="restored.png" badge={res && "output"} />
              {res?.quality && <ImagePanel title="Absolute error map" src={res.quality.error_map} filename="error_map.png" />}
            </div>
            {res?.clean_image && (
              <details className="mt-4 text-sm text-slate-600">
                <summary className="cursor-pointer">Show original (clean) image</summary>
                <img src={res.clean_image} alt="clean" className="mt-2 h-32 w-32 rounded-lg border" />
              </details>
            )}
          </Card>

          {res && (
            <Card title="System information">
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                <Stat label="Inference time" value={timing} hint={kind === "hard" ? `classifier ${res.timing_ms.classifier} + expert ${res.timing_ms.expert} ms` : "ONNX Runtime · CPU"} />
                {res.quality && <Stat label="PSNR corrupted → restored" value={`${res.quality.psnr_corrupted_db} → ${res.quality.psnr_restored_db} dB`} />}
                {kind === "hard" && <Stat label="Selected expert" value={res.expert} hint={`predicted: ${res.predicted}`} />}
                {kind === "soft" && <Stat label="Dominant branch" value={res.dominant} hint={`${(res.dominant_weight * 100).toFixed(1)}% · entropy ${res.entropy}/${res.max_entropy}`} />}
              </div>
              <p className="mt-3 text-sm text-slate-600"><span className="font-medium">Corruption settings:</span> {settingsText(res.corruption)}</p>
              {kind === "hard" && res.misrouted && <p className="mt-2 rounded-md bg-amber-50 px-3 py-2 text-sm text-amber-800">Classifier error: predicted “{res.predicted}” but the applied corruption was “{res.corruption.type}”.</p>}
            </Card>
          )}

          {res && kind === "hard" && (
            <Card title="Classifier probabilities" right={<span className="text-xs text-slate-500">mode: {res.mode}</span>}>
              <WeightBars data={res.probabilities} highlight={res.predicted} />
            </Card>
          )}

          {res && kind === "soft" && (
            <Card title="Routing weights (gate output)">
              <WeightBars data={res.weights} highlight={res.dominant} />
              <div className="mt-4">
                <div className="mb-1 text-xs text-slate-500">Contribution to the final image</div>
                <div className="flex h-6 overflow-hidden rounded-md text-[10px] text-white">
                  {Object.entries(res.weights).map(([k, v], i) => (
                    <div key={k} title={`${k}: ${(v * 100).toFixed(1)}%`} style={{ width: `${v * 100}%` }} className={["bg-slate-500", "bg-rose-500", "bg-sky-500", "bg-emerald-500"][i]}>
                      {v > 0.12 && <span className="px-1">{k.split(" ")[0]}</span>}
                    </div>
                  ))}
                </div>
              </div>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
