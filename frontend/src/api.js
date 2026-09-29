// All backend calls. Paths are relative: Vite proxy (dev) / nginx (Docker) forward /api to FastAPI.
export async function getHealth() {
  const r = await fetch("/api/health");
  if (!r.ok) throw new Error(`health ${r.status}`);
  return r.json();
}

export async function getSamples() {
  const r = await fetch("/api/samples");
  return r.ok ? (await r.json()).samples : [];
}

export async function post(path, { file, sample, fields }) {
  const fd = new FormData();
  if (file) fd.append("file", file, file.name || "image.png");
  else if (sample) fd.append("sample", sample);
  Object.entries(fields || {}).forEach(([k, v]) => fd.append(k, String(v)));
  const r = await fetch(path, { method: "POST", body: fd });
  const body = await r.json().catch(() => ({}));
  if (!r.ok) {
    const d = body.detail;
    throw new Error(typeof d === "string" ? d : Array.isArray(d) ? d.map((e) => e.msg).join("; ") : `Request failed (${r.status})`);
  }
  return body;
}

export function download(dataUrl, name) {
  const a = document.createElement("a");
  a.href = dataUrl;
  a.download = name;
  a.click();
}
