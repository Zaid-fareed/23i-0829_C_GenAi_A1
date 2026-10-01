import React, { useEffect, useRef, useState } from "react";
import { getSamples } from "../api.js";
import { Button, Segmented } from "./ui.jsx";

/**
 * Compact input picker (Stitch: Upload / Sample / Webcam tabs + a one-line summary of the chosen image).
 * Reports { file, sample, previewUrl, name } through onChange.
 */
export default function ImageSource({ value, onChange, webcam = false, sampleFilter = null }) {
  const [samples, setSamples] = useState([]);
  const [tab, setTab] = useState("sample");
  const [camOn, setCamOn] = useState(false);
  const [camErr, setCamErr] = useState("");
  const [over, setOver] = useState(false);
  const video = useRef(null);
  const stream = useRef(null);

  useEffect(() => { getSamples().then(setSamples).catch(() => {}); }, []);
  useEffect(() => () => stopCam(), []);

  const stopCam = () => {
    stream.current?.getTracks().forEach((t) => t.stop());
    stream.current = null;
    setCamOn(false);
  };

  const pickFile = (f) => {
    if (!f) return;
    onChange({ file: f, sample: null, previewUrl: URL.createObjectURL(f), name: f.name });
  };

  const startCam = async () => {
    setCamErr("");
    try {
      stream.current = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } });
      setCamOn(true);
      requestAnimationFrame(() => { if (video.current) video.current.srcObject = stream.current; });
    } catch (e) {
      setCamErr("Webcam unavailable: " + (e.message || e.name) + ". Use upload instead.");
    }
  };

  const snap = () => {
    const v = video.current;
    const c = document.createElement("canvas");
    c.width = v.videoWidth; c.height = v.videoHeight;
    c.getContext("2d").drawImage(v, 0, 0);
    c.toBlob((b) => { pickFile(new File([b], "webcam.png", { type: "image/png" })); stopCam(); }, "image/png");
  };

  const shown = sampleFilter ? samples.filter((s) => s.startsWith(sampleFilter)) : samples;
  const tabs = [["sample", "Sample"], ["upload", "Upload"], ...(webcam ? [["webcam", "Webcam"]] : [])];

  return (
    <div className="space-y-2.5">
      <Segmented options={tabs} value={tab} onChange={(k) => { setTab(k); if (k !== "webcam") stopCam(); }} />

      {tab === "upload" && (
        <label
          onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)}
          onDrop={(e) => { e.preventDefault(); setOver(false); pickFile(e.dataTransfer.files[0]); }}
          className={`flex cursor-pointer flex-col items-center justify-center gap-0.5 rounded-xl border-[1.5px] border-dashed px-3 py-4 text-center text-sm text-slate-600 transition ${over ? "border-brand-500 bg-brand-50" : "border-slate-300 bg-slate-50 hover:border-brand-500"}`}>
          <span className="font-medium">Drop an image or click to browse</span>
          <span className="font-mono text-[11px] text-slate-400">JPEG / PNG / WebP · max 10 MB</span>
          <input type="file" accept="image/jpeg,image/png,image/webp,image/bmp" className="hidden" onChange={(e) => pickFile(e.target.files[0])} />
        </label>
      )}

      {tab === "sample" && (
        shown.length ? (
          <div className="grid grid-cols-6 gap-1.5">
            {shown.map((s) => (
              <button key={s} title={s} onClick={() => onChange({ file: null, sample: s, previewUrl: `/api/samples/${s}`, name: s })}
                className={`aspect-square overflow-hidden rounded-lg border-2 ${value.sample === s ? "border-brand-600" : "border-transparent hover:border-slate-300"}`}>
                <img src={`/api/samples/${s}`} alt={s} className="h-full w-full object-cover" />
              </button>
            ))}
          </div>
        ) : <p className="text-sm text-slate-500">No samples on the server. Put images in <code className="font-mono">backend/samples</code>.</p>
      )}

      {tab === "webcam" && (
        <div className="space-y-2">
          {camOn ? (
            <>
              <video ref={video} autoPlay playsInline muted className="w-full rounded-xl bg-black" />
              <Button className="w-full" onClick={snap}>Capture photo</Button>
            </>
          ) : <Button variant="secondary" className="w-full" onClick={startCam}>Start webcam</Button>}
          {camErr && <p className="text-sm text-rose-600">{camErr}</p>}
        </div>
      )}

      {value.file && value.previewUrl && (  /* samples are highlighted in the grid; only uploads/webcam need a preview row */
        <div className="flex items-center gap-3 rounded-xl border border-slate-200 bg-slate-50 p-2">
          <img src={value.previewUrl} alt="selected" className="h-12 w-12 shrink-0 rounded-lg object-cover" />
          <div className="min-w-0">
            <div className="truncate font-mono text-xs text-slate-900">{value.name}</div>
            <div className="text-[11px] text-slate-400">selected input</div>
          </div>
        </div>
      )}
    </div>
  );
}
