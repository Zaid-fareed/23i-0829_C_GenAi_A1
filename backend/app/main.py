"""FastAPI backend: validates uploads, applies optional runtime corruption, runs ONNX models, returns
images (PNG data URLs) plus routing / timing information for the four workspaces."""
import os
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from . import corruptions as C
from .imaging import decode_upload, error_map, psnr, to_data_url, to_tensor
from .registry import ModelRegistry

CLASSES = ["clean", "salt_pepper", "blur", "occlusion"]
SPECIALISTS = {"salt_pepper": "specialist_salt_pepper", "blur": "specialist_blur", "occlusion": "specialist_occlusion"}
SAMPLES_DIR = Path(os.environ.get("SAMPLES_DIR", Path(__file__).resolve().parents[1] / "samples"))

app = FastAPI(title="GenAI Assignment 1 - Restoration & Face-to-Sketch API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
registry = ModelRegistry()


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


# ----------------------------------------------------------------------------- input handling
async def load_input(file: UploadFile | None, sample: str | None):
    if file is not None and file.filename:
        return decode_upload(await file.read(), file.content_type), file.filename
    if sample:
        p = (SAMPLES_DIR / Path(sample).name)  # .name blocks path traversal
        if not p.is_file():
            raise HTTPException(404, f"Sample '{sample}' not found.")
        return decode_upload(p.read_bytes(), None), p.name
    raise HTTPException(400, "Provide an image file or a sample name.")


def _check(name, v, lo, hi):
    if not (lo <= v <= hi):
        raise HTTPException(422, f"{name} must be between {lo} and {hi}.")


def corruption_from_form(corruption, severity, seed, sp_p, blur_kernel, blur_sigma, occ_coverage, occ_rects):
    """Returns spec dict or None (no corruption: image is used as-is, e.g. an already-corrupted upload)."""
    if corruption == "none":
        return None
    if corruption not in C.TYPES[1:]:
        raise HTTPException(422, f"corruption must be one of none, {', '.join(C.TYPES[1:])}.")
    if severity in ("low", "medium", "high"):
        rng = np.random.default_rng(seed)
        return C.make_spec(corruption, rng, severity)
    if severity != "custom":
        raise HTTPException(422, "severity must be low, medium, high or custom.")
    _check("sp_p", sp_p, 0.0, 0.5)
    _check("blur_sigma", blur_sigma, 0.1, 5.0)
    _check("occ_coverage", occ_coverage, 0.05, 0.6)
    _check("occ_rects", occ_rects, 1, 5)
    if blur_kernel % 2 == 0 or not 3 <= blur_kernel <= 15:
        raise HTTPException(422, "blur_kernel must be an odd number between 3 and 15.")
    return C.custom_spec(corruption, seed, sp_p=sp_p, blur_kernel=blur_kernel, blur_sigma=blur_sigma,
                         occ_coverage=occ_coverage, occ_rects=occ_rects)


def describe(spec):
    if spec is None:
        return {"applied": False}
    d = {k: v for k, v in spec.items() if k not in ("rects", "noise_seed")}
    d["applied"] = True
    if "coverage" in d:
        d["coverage"] = round(d["coverage"], 4)
    return d


def prepare(im, spec):
    clean = to_tensor(im)
    x = C.apply_spec(clean, spec) if spec else clean
    extra = {}
    if spec:  # the upload is assumed to be the clean target -> quality numbers + error map are meaningful
            extra = {"clean_image": to_data_url(clean)}
    return clean, x.astype(np.float32), extra


def quality(clean, x, out, spec):
    if not spec:
        return None
    return {"psnr_corrupted_db": round(psnr(clean, x), 2), "psnr_restored_db": round(psnr(clean, out), 2),
            "error_map": to_data_url(error_map(out, clean))}


# ----------------------------------------------------------------------------- endpoints
@app.get("/api/health")
def health():
    return {"status": "ok", **registry.status()}


@app.get("/api/samples")
def samples():
    files = sorted(p.name for p in SAMPLES_DIR.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")) \
        if SAMPLES_DIR.exists() else []
    return {"samples": files}


@app.get("/api/samples/{name}")
def sample_file(name: str):
    p = SAMPLES_DIR / Path(name).name
    if not p.is_file():
        raise HTTPException(404, "Sample not found.")
    return FileResponse(p)


@app.post("/api/restore/universal")
async def restore_universal(file: UploadFile | None = File(None), sample: str | None = Form(None),
                            corruption: str = Form("none"), severity: str = Form("medium"), seed: int = Form(42),
                            sp_p: float = Form(0.08), blur_kernel: int = Form(5), blur_sigma: float = Form(1.5),
                            occ_coverage: float = Form(0.2), occ_rects: int = Form(2)):
    registry.require("universal")
    im, name = await load_input(file, sample)
    spec = corruption_from_form(corruption, severity, seed, sp_p, blur_kernel, blur_sigma, occ_coverage, occ_rects)
    clean, x, extra = prepare(im, spec)
    (out,), ms = registry.run("universal", {"image": x[None]})
    out = np.clip(out[0], 0, 1)
    return {"filename": name, "input_image": to_data_url(x), **extra, "output_image": to_data_url(out),
            "corruption": describe(spec), "inference_ms": round(ms, 2), "quality": quality(clean, x, out, spec)}


@app.post("/api/restore/hard")
async def restore_hard(file: UploadFile | None = File(None), sample: str | None = Form(None),
                       mode: str = Form("predicted"),
                       corruption: str = Form("none"), severity: str = Form("medium"), seed: int = Form(42),
                       sp_p: float = Form(0.08), blur_kernel: int = Form(5), blur_sigma: float = Form(1.5),
                       occ_coverage: float = Form(0.2), occ_rects: int = Form(2)):
    registry.require("hard")
    if mode not in ("predicted", "oracle"):
        raise HTTPException(422, "mode must be 'predicted' or 'oracle'.")
    im, name = await load_input(file, sample)
    spec = corruption_from_form(corruption, severity, seed, sp_p, blur_kernel, blur_sigma, occ_coverage, occ_rects)
    if mode == "oracle" and spec is None:
        raise HTTPException(400, "Oracle routing needs a known corruption: choose a corruption type to apply.")
    clean, x, extra = prepare(im, spec)

    (logits,), t_cls = registry.run("classifier", {"image": x[None]})
    probs = softmax(logits[0].astype(np.float64))
    predicted = CLASSES[int(probs.argmax())]
    routed = spec["type"] if mode == "oracle" else predicted
    if routed == "clean":  # identity bypass: no restoration expert is run
        out, t_exp, expert = x, 0.0, "identity (bypass)"
    else:
        (o,), t_exp = registry.run(SPECIALISTS[routed], {"image": x[None]})
        out, expert = np.clip(o[0], 0, 1), f"{routed} specialist"
    actual = spec["type"] if spec else None
    return {"filename": name, "input_image": to_data_url(x), **extra, "output_image": to_data_url(out),
            "corruption": describe(spec), "mode": mode,
            "probabilities": {c: round(float(p), 5) for c, p in zip(CLASSES, probs)},
            "predicted": predicted, "routed_to": routed, "expert": expert,
            "misrouted": (actual is not None and predicted != actual),
            "timing_ms": {"classifier": round(t_cls, 2), "expert": round(t_exp, 2), "total": round(t_cls + t_exp, 2)},
            "quality": quality(clean, x, out, spec)}


@app.post("/api/restore/soft")
async def restore_soft(file: UploadFile | None = File(None), sample: str | None = Form(None),
                       corruption: str = Form("none"), severity: str = Form("medium"), seed: int = Form(42),
                       sp_p: float = Form(0.08), blur_kernel: int = Form(5), blur_sigma: float = Form(1.5),
                       occ_coverage: float = Form(0.2), occ_rects: int = Form(2)):
    registry.require("soft")
    im, name = await load_input(file, sample)
    spec = corruption_from_form(corruption, severity, seed, sp_p, blur_kernel, blur_sigma, occ_coverage, occ_rects)
    clean, x, extra = prepare(im, spec)
    (out, w), ms = registry.run("soft_moe", {"image": x[None]})
    out, w = np.clip(out[0], 0, 1), w[0].astype(np.float64)
    labels = ["identity (clean)", "salt_pepper expert", "blur expert", "occlusion expert"]
    weights = {l: round(float(v), 5) for l, v in zip(labels, w)}
    entropy = float(-(w * np.log(w + 1e-12)).sum())
    return {"filename": name, "input_image": to_data_url(x), **extra, "output_image": to_data_url(out),
            "corruption": describe(spec), "weights": weights,
            "dominant": labels[int(w.argmax())], "dominant_weight": round(float(w.max()), 5),
            "entropy": round(entropy, 4), "max_entropy": round(float(np.log(4)), 4),
            "inference_ms": round(ms, 2), "quality": quality(clean, x, out, spec)}


@app.post("/api/sketch")
async def face_to_sketch(file: UploadFile | None = File(None), sample: str | None = Form(None),
                         style: int = Form(1)):
    registry.require("sketch")
    if style not in (1, 2, 3):
        raise HTTPException(422, "style must be 1, 2 or 3.")
    im, name = await load_input(file, sample)
    photo = to_tensor(im)
    feeds = {"photo": (photo * 2 - 1)[None].astype(np.float32), "style": np.array([style - 1], dtype=np.int64)}
    (o,), ms = registry.run("sketch", feeds)
    sketch = (np.clip(o[0, 0], -1, 1) + 1) / 2
    return {"filename": name, "style": f"Style {style}", "input_image": to_data_url(photo),
            "output_image": to_data_url(sketch), "inference_ms": round(ms, 2)}
