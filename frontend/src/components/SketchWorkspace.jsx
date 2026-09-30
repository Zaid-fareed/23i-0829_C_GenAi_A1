import React, { useState } from "react";
import { download, post } from "../api.js";
import ImageSource from "./ImageSource.jsx";
import { Button, Card, ErrorBox, ImagePanel, Mono, Pill, Segmented, Stat } from "./ui.jsx";

const STYLES = [
  { id: 1, name: "Style 1", note: "FS2K sketch style category 1" },
  { id: 2, name: "Style 2", note: "FS2K sketch style category 2" },
  { id: 3, name: "Style 3", note: "FS2K sketch style category 3" },
];

export default function SketchWorkspace() {
  const [src, setSrc] = useState({ file: null, sample: null, previewUrl: null, name: "" });
  const [style, setStyle] = useState(1);
  const [fit, setFit] = useState("crop");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [res, setRes] = useState(null);
  const [recent, setRecent] = useState([]);  // session-only history, like the Stitch "recent runs" card

  const run = async () => {
    setBusy(true); setErr("");
    try {
      const r = await post("/api/sketch", { file: src.file, sample: src.sample, fields: { style, fit } });
      setRes(r);
      setRecent((h) => [{ name: src.name, style: r.style, ms: r.inference_ms, img: r.output_image, at: new Date() }, ...h].slice(0, 3));
    } catch (e) { setErr(e.message); setRes(null); }
    finally { setBusy(false); }
  };

  return (
    <div className="space-y-6">
      <header>
        <h2 className="text-[28px] font-bold leading-9 tracking-tight text-slate-900">Face-to-Sketch Generator</h2>
        <p className="mt-1 max-w-3xl text-slate-600">A style-conditioned cGAN (U-Net generator, learned style embedding) turns a face photograph into a sketch in one of three FS2K styles.</p>
      </header>
      <div className="grid gap-6 xl:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
        <div className="space-y-6">
          <Card step={1} title="Portrait input" tag="Photograph">
            <ImageSource value={src} onChange={(s) => { setSrc(s); setRes(null); }} webcam />
          </Card>
          <Card step={2} title="Synthesis style" tag="Condition">
            <div className="space-y-2">
              {STYLES.map((s) => (
                <button key={s.id} onClick={() => setStyle(s.id)}
                  className={`flex w-full items-center justify-between rounded-xl border px-4 py-3 text-left transition ${style === s.id ? "border-brand-500 bg-brand-50/70" : "border-slate-200 hover:bg-slate-50"}`}>
                  <span><span className="block font-semibold text-slate-900">{s.name}</span><span className="font-mono text-xs text-slate-500">{s.note} · embedding #{s.id - 1}</span></span>
                  <span className={`grid h-5 w-5 place-items-center rounded-full border ${style === s.id ? "border-brand-600 bg-brand-600 text-[11px] text-white" : "border-slate-300"}`}>{style === s.id && "✓"}</span>
                </button>
              ))}
            </div>
          </Card>
          <Card step={3} title="Framing" tag="Preprocessing">
            <Segmented options={[["crop", "Crop to square"], ["pad", "Pad"], ["stretch", "Stretch"]]} value={fit} onChange={setFit} />
            <p className="mt-2 text-xs text-slate-400">The model was trained on near-square head-and-shoulders photos. Crop works best for tall or wide photos; the left panel shows exactly what the model sees.</p>
          </Card>
          <Button className="w-full !py-3.5 text-base" onClick={run} disabled={!(src.file || src.sample) || busy}>
            {busy ? "Generating…" : "✦  Generate sketch"}
            {res && !busy && <Mono className="rounded-full bg-white/20 px-2 py-0.5 text-xs">~{res.inference_ms} ms</Mono>}
          </Button>
          <ErrorBox msg={err} />
        </div>

        <div className="space-y-6">
          <Card title="Synthesis viewport" right={res ? <Pill tone="ok">Done · {res.inference_ms} ms</Pill> : <Pill tone="warn">Waiting for run</Pill>}>
            <div className="grid grid-cols-2 gap-4">
              <ImagePanel title="Original photograph" src={res?.input_image} badge={res && "input · 128×128"} />
              <ImagePanel title={res ? `Generated sketch · ${res.style}` : "Generated sketch"} src={res?.output_image} badge={res && "grayscale"} />
            </div>
            {res && (
              <div className="mt-4 flex flex-wrap gap-2">
                <Button onClick={() => download(res.output_image, `sketch_style${style}.png`)}>↓ Export sketch (PNG)</Button>
                <Button variant="secondary" onClick={() => download(res.input_image, "photo_128.png")}>↓ Photo (PNG)</Button>
              </div>
            )}
          </Card>
          {res && (
            <Card title="System information" right={<span className="font-mono text-[10px] uppercase tracking-widest text-slate-500">ONNX Runtime · CPU</span>}>
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
                <Stat label="Inference time" value={`${res.inference_ms} ms`} hint="generator forward pass" />
                <Stat label="Style condition" value={res.style} hint={`embedding #${style - 1}`} />
                <Stat label="Output" value="128 × 128" hint="8-bit grayscale" />
              </div>
            </Card>
          )}
          {recent.length > 0 && (
            <Card title="Recent synthesis runs" right={<span className="font-mono text-[10px] uppercase tracking-widest text-slate-500">{recent.length} in this session</span>}>
              <div className="grid grid-cols-3 gap-3">
                {recent.map((r, i) => (
                  <div key={i} className="min-w-0">
                    <img src={r.img} alt="" className="aspect-square w-full rounded-lg border border-slate-200 object-cover" />
                    <div className="mt-1 truncate text-sm font-medium text-slate-800">{r.name || "image"}</div>
                    <div className="flex justify-between font-mono text-[11px] text-slate-500"><span>{r.style}</span><span>{r.ms} ms</span></div>
                  </div>
                ))}
              </div>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
