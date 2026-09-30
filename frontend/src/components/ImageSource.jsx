import React, { useEffect, useRef, useState } from "react";
import { getSamples } from "../api.js";
import { Button, Segmented } from "./ui.jsx";

/**
 * Input picker (Stitch: Upload / Sample / Webcam segmented control).
 * Reports { file, sample, previewUrl, name } through onChange.
 */
export default function ImageSource({ value, onChange, webcam = false }) {
  const [samples, setSamples] = useState([]);
  const [tab, setTab] = useState("upload");
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

  const tabs = [["upload", "Upload"], ["sample", "Sample"], ...(webcam ? [["webcam", "Webcam"]] : [])];

  return (
    <div className="space-y-3">
      <Segmented options={tabs} value={tab} onChange={(k) => { setTab(k); if (k !== "webcam") stopCam(); }} />

      {tab === "upload" && (
        <label
          onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)}
          onDrop={(e) => { e.preventDefault(); setOver(false); pickFile(e.dataTransfer.files[0]); }}
          className={`flex cursor-pointer flex-col items-center justify-center gap-1 rounded-xl border-[1.5px] border-dashed px-4 py-6 text-sm text-slate-600 transition ${over ? "border-brand-500 bg-brand-50" : "border-slate-300 bg-slate-50 hover:border-brand-500"}`}>
          <span className="font-medium">Drag &amp; drop an image, or click to browse</span>
          <span className="font-mono text-[11px] text-slate-400">JPEG / PNG / WebP · max 10 MB</span>
          <input type="file" accept="image/jpeg,image/png,image/webp,image/bmp" className="hidden" onChange={(e) => pickFile(e.target.files[0])} />
        </label>
      )}

      {tab === "sample" && (
        samples.length ? (
          <div className="flex flex-wrap gap-2">
            {samples.map((s) => (
              <button key={s} onClick={() => onChange({ file: null, sample: s, previewUrl: `/api/samples/${s}`, name: s })}
                className={`h-16 w-16 overflow-hidden rounded-lg border-2 ${value.sample === s ? "border-brand-600" : "border-transparent"}`}>
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
              <video ref={video} autoPlay playsInline muted className="w-full max-w-sm rounded-xl bg-black" />
              <Button onClick={snap}>Capture photo</Button>
            </>
          ) : <Button variant="secondary" onClick={startCam}>Start webcam</Button>}
          {camErr && <p className="text-sm text-rose-600">{camErr}</p>}
        </div>
      )}

      {value.previewUrl && (
        <div className="relative overflow-hidden rounded-xl border border-slate-200 bg-slate-100">
          <img src={value.previewUrl} alt="selected" className="mx-auto max-h-56 w-full object-contain" />
          <span className="absolute bottom-2 left-2 max-w-[80%] truncate rounded-md bg-slate-900/70 px-2 py-0.5 font-mono text-[11px] text-white">{value.name}</span>
        </div>
      )}
    </div>
  );
}
