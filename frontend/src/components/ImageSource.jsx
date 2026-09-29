import React, { useEffect, useRef, useState } from "react";
import { getSamples } from "../api.js";
import { Button } from "./ui.jsx";

/**
 * Input picker: upload a file, choose a server-side sample, or (optionally) capture from the webcam.
 * Reports { file, sample, previewUrl } through onChange.
 */
export default function ImageSource({ value, onChange, webcam = false }) {
  const [samples, setSamples] = useState([]);
  const [tab, setTab] = useState("upload");
  const [camOn, setCamOn] = useState(false);
  const [camErr, setCamErr] = useState("");
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
    onChange({ file: f, sample: null, previewUrl: URL.createObjectURL(f) });
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
    c.toBlob((b) => {
      const f = new File([b], "webcam.png", { type: "image/png" });
      pickFile(f);
      stopCam();
    }, "image/png");
  };

  const tabs = [["upload", "Upload"], ["sample", "Sample"], ...(webcam ? [["webcam", "Webcam"]] : [])];

  return (
    <div>
      <div className="mb-3 inline-flex rounded-lg bg-slate-100 p-1 text-sm">
        {tabs.map(([k, l]) => (
          <button key={k} onClick={() => { setTab(k); if (k !== "webcam") stopCam(); }}
            className={`rounded-md px-3 py-1 ${tab === k ? "bg-white font-medium shadow-sm" : "text-slate-600"}`}>{l}</button>
        ))}
      </div>

      {tab === "upload" && (
        <label className="flex cursor-pointer flex-col items-center justify-center gap-1 rounded-xl border-2 border-dashed border-slate-300 bg-slate-50 px-4 py-6 text-sm text-slate-600 hover:border-brand-500">
          <span className="font-medium">Click to choose an image</span>
          <span className="text-xs text-slate-400">JPEG / PNG / WebP, up to 10 MB</span>
          <input type="file" accept="image/jpeg,image/png,image/webp,image/bmp" className="hidden" onChange={(e) => pickFile(e.target.files[0])} />
        </label>
      )}

      {tab === "sample" && (
        samples.length ? (
          <div className="flex flex-wrap gap-2">
            {samples.map((s) => (
              <button key={s} onClick={() => onChange({ file: null, sample: s, previewUrl: `/api/samples/${s}` })}
                className={`h-16 w-16 overflow-hidden rounded-lg border-2 ${value.sample === s ? "border-brand-600" : "border-transparent"}`}>
                <img src={`/api/samples/${s}`} alt={s} className="h-full w-full object-cover" />
              </button>
            ))}
          </div>
        ) : <p className="text-sm text-slate-500">No samples on the server. Put images in <code>backend/samples</code>.</p>
      )}

      {tab === "webcam" && (
        <div className="space-y-2">
          {camOn ? (
            <>
              <video ref={video} autoPlay playsInline muted className="w-full max-w-sm rounded-xl bg-black" />
              <Button onClick={snap}>Capture photo</Button>
            </>
          ) : <Button variant="secondary" onClick={startCam}>Start webcam</Button>}
          {camErr && <p className="text-sm text-red-600">{camErr}</p>}
        </div>
      )}
    </div>
  );
}
