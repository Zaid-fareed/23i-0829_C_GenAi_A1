import React, { useState } from "react";
import { post } from "../api.js";
import ImageSource from "./ImageSource.jsx";
import CorruptionControls, { DEFAULT_CORRUPTION } from "./CorruptionControls.jsx";
import { Button, Card, ErrorBox, ImagePanel, Mono, Pill, Section, Segmented, Stat, TileRow, WeightBars, Workbench } from "./ui.jsx";

const KINDS = {
  universal: {
    title: "Universal Restoration",
    blurb: "One autoencoder restores clean, noisy, blurred and occluded images without being told which.",
    path: "/api/restore/universal",
    reserve: 340,
  },
  hard: {
    title: "Hard-Routed Restoration",
    blurb: "A classifier picks one specialist per image; clean images bypass restoration.",
    path: "/api/restore/hard",
    reserve: 500,
  },
  soft: {
    title: "Soft Mixture-of-Experts",
    blurb: "A gate weights all branches (identity + 3 experts); the output is their weighted sum.",
    path: "/api/restore/soft",
    reserve: 500,
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
  const [showClean, setShowClean] = useState(false);

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

  const controls = (
    <>
      <Section n="01" title="Input image" tag="Source">
        <ImageSource value={src} onChange={(s) => { setSrc(s); setRes(null); }} sampleFilter="pet_" />
      </Section>
      <Section n="02" title="Runtime corruption" tag="Synthesizer">
        <CorruptionControls value={cor} onChange={setCor} />
      </Section>
      {kind === "hard" && (
        <Section n="03" title="Routing mode">
          <Segmented options={[["predicted", "Predicted"], ["oracle", "Oracle (true label)"]]} value={mode} onChange={setMode} />
        </Section>
      )}
      <ErrorBox msg={err} />
    </>
  );

  const footer = (
    <Button className="w-full !py-3 text-base" onClick={run} disabled={!ready || busy}>
      {busy ? "Running…" : "▷  Run model"}
      {res && !busy && <Mono className="rounded-full bg-white/20 px-2 py-0.5 text-xs">~{ms} ms</Mono>}
    </Button>
  );

  return (
    <Workbench controls={controls} footer={footer}>
      <div className="mb-2 flex items-start justify-between gap-2">
        <div className="min-w-0">
          <h2 className="text-xl font-bold leading-7 tracking-tight text-slate-900">{cfg.title}</h2>
          <p className="text-sm text-slate-500">{cfg.blurb}</p>
        </div>
        <span className="shrink-0">{res ? <Pill tone="ok">Done · {ms} ms</Pill> : <Pill tone="warn">{ready ? "Ready to run" : "Pick an image"}</Pill>}</span>
      </div>

      <TileRow cols={3} reserve={cfg.reserve}>
        <ImagePanel index={1} title="Model input" src={res?.input_image} filename="input.png" badge={res && (res.corruption.applied ? "corrupted" : "as uploaded")} hint="Pick an image, choose a corruption, press Run" />
        <ImagePanel index={2} title="Restored output" src={res?.output_image} filename="restored.png" badge={res && "output"} hint="The restored image appears here" />
        <ImagePanel index={3} title="Absolute error map" src={res?.quality?.error_map} filename="error_map.png" badge={res?.quality && "|restored − clean|"}
          hint={res && !res.quality ? "Needs a clean reference: apply a corruption" : "Where the restoration differs from the original"} />
      </TileRow>

      {res && kind === "hard" && (
        <Card className="mt-2" title="Classifier probabilities" right={<Mono className="text-xs text-slate-500">mode: {res.mode} · argmax</Mono>}>
          <WeightBars data={res.probabilities} selected={res.predicted} selectedLabel="Predicted" cols={2} />
          {res.misrouted && <p className="mt-2 rounded-md bg-amber-50 px-3 py-2 text-sm text-amber-800">Classifier error: predicted “{res.predicted}” but the applied corruption was “{res.corruption.type}”.</p>}
        </Card>
      )}

      {res && kind === "soft" && (
        <Card className="mt-2" title="Routing weights (gate output)">
          <WeightBars data={res.weights} selected={res.dominant} selectedLabel="Dominant" cols={2} />
          <div className="mt-2 flex h-5 overflow-hidden rounded-md text-[10px] text-white">
            {Object.entries(res.weights).map(([k, v], i) => (
              <div key={k} title={`${k}: ${(v * 100).toFixed(1)}%`} style={{ width: `${v * 100}%` }} className={STRIP[i]}>
                {v > 0.12 && <span className="px-1 font-mono">{k.split(" ")[0]}</span>}
              </div>
            ))}
          </div>
        </Card>
      )}

      {res ? (
        <div className="mt-3">
          <div className="grid grid-cols-2 gap-2 xl:grid-cols-4">
            <Stat label="Inference" value={`${ms} ms`} hint={kind === "hard" ? `clf ${res.timing_ms.classifier} · exp ${res.timing_ms.expert} ms` : "ONNX Runtime · CPU"} />
            {res.quality
              ? <Stat label="PSNR before → after" value={`${res.quality.psnr_corrupted_db} → ${res.quality.psnr_restored_db}`} hint={`${(res.quality.psnr_restored_db - res.quality.psnr_corrupted_db) >= 0 ? "+" : ""}${(res.quality.psnr_restored_db - res.quality.psnr_corrupted_db).toFixed(1)} dB`} />
              : <Stat label="PSNR" value="n/a" hint="no clean reference" />}
            {kind === "hard" && <Stat label="Selected expert" value={res.expert} hint={`predicted: ${res.predicted}`} />}
            {kind === "hard" && <Stat label="Confidence" value={topProb.toFixed(3)} hint={topProb > 0.9 ? "high" : "low"} />}
            {kind === "soft" && <Stat label="Dominant" value={res.dominant} hint={`${(res.dominant_weight * 100).toFixed(1)}%`} />}
            {kind === "soft" && <Stat label="Entropy" value={`${res.entropy}/${res.max_entropy}`} hint={res.entropy > 0.6 * res.max_entropy ? "distributed" : "one expert dominates"} />}
            {kind === "universal" && <Stat label="Corruption" value={res.corruption.applied ? res.corruption.type : "none"} hint={res.corruption.applied ? res.corruption.severity : "as uploaded"} />}
            {kind === "universal" && <Stat label="Model" value="1 autoencoder" hint="no routing" />}
          </div>
          {showClean && res.clean_image && <img src={res.clean_image} alt="clean" className="mt-2 h-24 w-24 rounded-lg border" />}
          <div className="mt-2 flex flex-wrap items-center justify-between gap-2 rounded-lg bg-slate-50 px-3 py-2 text-xs text-slate-600">
            <span className="font-mono">PARAMS: {settingsText(res.corruption)}</span>
            {res.clean_image && <button onClick={() => setShowClean(!showClean)} className="font-mono text-slate-500 hover:text-slate-800">{showClean ? "hide original" : "show original"}</button>}
            <button onClick={copyJson} className="font-mono font-semibold text-brand-600 hover:text-brand-700">{copied ? "Copied ✓" : "Copy JSON"}</button>
          </div>
        </div>
      ) : (
        <p className="mt-4 rounded-lg bg-slate-50 px-4 py-3 text-sm text-slate-500">
          Press <strong>Run model</strong> (bottom of the left panel). Timing, PSNR{kind === "hard" ? ", classifier probabilities and the selected expert" : kind === "soft" ? " and the expert weights" : ""} will appear here.
        </p>
      )}
    </Workbench>
  );
}
