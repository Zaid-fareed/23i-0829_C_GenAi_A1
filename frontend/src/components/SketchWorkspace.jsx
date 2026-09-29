import React, { useState } from "react";
import { post } from "../api.js";
import ImageSource from "./ImageSource.jsx";
import { Button, Card, ErrorBox, ImagePanel, Stat } from "./ui.jsx";

export default function SketchWorkspace() {
  const [src, setSrc] = useState({ file: null, sample: null, previewUrl: null });
  const [style, setStyle] = useState(1);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [res, setRes] = useState(null);

  const run = async () => {
    setBusy(true); setErr("");
    try { setRes(await post("/api/sketch", { file: src.file, sample: src.sample, fields: { style } })); }
    catch (e) { setErr(e.message); setRes(null); }
    finally { setBusy(false); }
  };

  return (
    <div className="space-y-6">
      <header>
        <h2 className="text-2xl font-semibold text-slate-900">Face-to-Sketch Generator</h2>
        <p className="mt-1 max-w-3xl text-slate-600">A style-conditioned cGAN (U-Net generator) turns a face photograph into a sketch in one of three FS2K artist styles.</p>
      </header>
      <div className="grid gap-6 lg:grid-cols-[360px_1fr]">
        <div className="space-y-6">
          <Card title="1 · Photograph"><ImageSource value={src} onChange={(s) => { setSrc(s); setRes(null); }} webcam />
            {src.previewUrl && <img src={src.previewUrl} alt="selected" className="mt-4 h-28 w-28 rounded-lg border object-cover" />}</Card>
          <Card title="2 · Sketch style">
            <div className="grid grid-cols-3 gap-1">
              {[1, 2, 3].map((s) => (
                <button key={s} onClick={() => setStyle(s)} className={`rounded-md border px-2 py-2 text-sm ${style === s ? "border-brand-600 bg-brand-50 font-medium text-brand-700" : "border-slate-300 text-slate-600"}`}>Style {s}</button>
              ))}
            </div>
          </Card>
          <Button className="w-full" onClick={run} disabled={!(src.file || src.sample) || busy}>{busy ? "Generating…" : "Generate sketch"}</Button>
          <ErrorBox msg={err} />
        </div>
        <div className="space-y-6">
          <Card title="Result">
            <div className="grid grid-cols-2 gap-4">
              <ImagePanel title="Original photograph" src={res?.input_image} />
              <ImagePanel title={res ? `Generated sketch (${res.style})` : "Generated sketch"} src={res?.output_image} filename={res ? `sketch_style${style}.png` : undefined} />
            </div>
          </Card>
          {res && <Card title="System information"><div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
            <Stat label="Inference time" value={`${res.inference_ms} ms`} hint="ONNX Runtime · CPU" />
            <Stat label="Style condition" value={res.style} />
            <Stat label="Output size" value="128 × 128" />
          </div></Card>}
        </div>
      </div>
    </div>
  );
}
