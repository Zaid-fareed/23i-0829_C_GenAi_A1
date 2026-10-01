import React, { useState } from "react";
import { download, post } from "../api.js";
import ImageSource from "./ImageSource.jsx";
import { Button, Mono, Pill, Section, Segmented, Stat, TileRow, ImagePanel, ErrorBox, Workbench } from "./ui.jsx";

const STYLES = [[1, "Style 1"], [2, "Style 2"], [3, "Style 3"]];

export default function SketchWorkspace() {
  const [src, setSrc] = useState({ file: null, sample: null, previewUrl: null, name: "" });
  const [style, setStyle] = useState(1);
  const [fit, setFit] = useState("crop");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [res, setRes] = useState(null);
  const [recent, setRecent] = useState([]);  // session-only history (last 3 runs)

  const run = async () => {
    setBusy(true); setErr("");
    try {
      const r = await post("/api/sketch", { file: src.file, sample: src.sample, fields: { style, fit } });
      setRes(r);
      setRecent((h) => [{ name: src.name, style: r.style, ms: r.inference_ms, img: r.output_image }, ...h].slice(0, 4));
    } catch (e) { setErr(e.message); setRes(null); }
    finally { setBusy(false); }
  };
  const ready = src.file || src.sample;

  const controls = (
    <>
      <Section n="01" title="Portrait input" tag="Photograph">
        <ImageSource value={src} onChange={(s) => { setSrc(s); setRes(null); }} webcam sampleFilter="face_" />
      </Section>
      <Section n="02" title="Sketch style" tag="Condition">
        <Segmented options={STYLES.map(([k, l]) => [k, l])} value={style} onChange={setStyle} />
        <p className="mt-1 text-xs text-slate-400">FS2K style category, given to the generator as an embedding.</p>
      </Section>
      <Section n="03" title="Framing" tag="Preprocessing">
        <Segmented options={[["crop", "Crop"], ["pad", "Pad"], ["stretch", "Stretch"]]} value={fit} onChange={setFit} />
        <p className="mt-1 text-xs text-slate-400">Crop suits tall/wide photos; the left tile shows what the model sees.</p>
      </Section>
      <ErrorBox msg={err} />
    </>
  );

  const footer = (
    <Button className="w-full !py-3 text-base" onClick={run} disabled={!ready || busy}>
      {busy ? "Generating…" : "✦  Generate sketch"}
      {res && !busy && <Mono className="rounded-full bg-white/20 px-2 py-0.5 text-xs">~{res.inference_ms} ms</Mono>}
    </Button>
  );

  return (
    <Workbench controls={controls} footer={footer}>
      <div className="mb-3 flex flex-wrap items-start justify-between gap-2">
        <div className="min-w-0">
          <h2 className="text-xl font-bold leading-7 tracking-tight text-slate-900">Face-to-Sketch Generator</h2>
          <p className="max-w-2xl text-sm text-slate-500">A style-conditioned cGAN (U-Net generator) turns a face photograph into a sketch in one of three styles.</p>
        </div>
        {res ? <Pill tone="ok">Done · {res.inference_ms} ms</Pill> : <Pill tone="warn">{ready ? "Ready to generate" : "Pick a photo"}</Pill>}
      </div>

      <TileRow cols={2} reserve={525}>
        <ImagePanel title="Original photograph" src={res?.input_image} filename="photo_128.png" badge={res && "model input · 128×128"} hint="Pick a photo and press Generate" />
        <ImagePanel title={res ? `Generated sketch · ${res.style}` : "Generated sketch"} src={res?.output_image} filename={`sketch_style${style}.png`} badge={res && "grayscale"} hint="The sketch appears here" />
      </TileRow>

      {res && (
        <div className="mt-2 flex flex-wrap items-center gap-2">
          <Button onClick={() => download(res.output_image, `sketch_style${style}.png`)}>↓ Download sketch (PNG)</Button>
          <Button variant="secondary" onClick={() => download(res.input_image, "photo_128.png")}>↓ Photo (PNG)</Button>
        </div>
      )}

      {res && (
        <div className="mt-2 grid grid-cols-2 gap-2 xl:grid-cols-4">
          <Stat label="Inference" value={`${res.inference_ms} ms`} hint="ONNX Runtime · CPU" />
          <Stat label="Style condition" value={res.style} hint={`embedding #${style - 1}`} />
          <Stat label="Framing" value={res.fit} hint="preprocessing" />
          <Stat label="Output" value="128 × 128" hint="8-bit grayscale" />
        </div>
      )}

      {recent.length > 0 && (
        <div className="mt-2">
          <div className="mb-1 font-mono text-[10px] font-semibold uppercase tracking-widest text-slate-500">Recent runs · this session</div>
          <div className="flex gap-2">
            {recent.map((r, i) => (
              <div key={i} className="w-20 shrink-0">
                <img src={r.img} alt="" className="aspect-square w-full rounded-lg border border-slate-200 object-cover" />
                <div className="font-mono text-[10px] leading-tight text-slate-500">{r.style.replace("Style ", "S")} · {Math.round(r.ms)} ms</div>
              </div>
            ))}
          </div>
        </div>
      )}
      {!res && <p className="mt-4 rounded-lg bg-slate-50 px-4 py-3 text-sm text-slate-500">Press <strong>Generate sketch</strong> (bottom of the left panel). The photo and sketch appear side by side here with a download button.</p>}
    </Workbench>
  );
}
