"""How much colour do the restoration models keep? (colourfulness = mean of max-min over the RGB channels)

Runs the ONNX models on salt-and-pepper TEST images (corrupted with the backend's numpy corruptions) and reports
each model's colourfulness as a % of the clean image's. Use it to compare model versions:

  PYTHONPATH=backend python scripts/check_colour_preservation.py --onnx_dir onnx_models
  PYTHONPATH=backend python scripts/check_colour_preservation.py --onnx_dir onnx_models_v2 --universal onnx_models/task1_universal_dae.onnx
"""
import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from app.corruptions import apply_spec

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--onnx_dir", default=str(ROOT / "onnx_models"))
ap.add_argument("--universal", default=None, help="Task 1 model path (default: <onnx_dir>/task1_universal_dae.onnx)")
ap.add_argument("--images", default=str(ROOT / "data/oxford-iiit-pet/images"))
ap.add_argument("--n", type=int, default=200)
a = ap.parse_args()
d = Path(a.onnx_dir)

entries = [e for e in json.load(open(ROOT / "manifests/test_manifest.json"))["entries"] if e["type"] == "salt_pepper"][:a.n]
models = {"universal": a.universal or d / "task1_universal_dae.onnx", "specialist": d / "task2_specialist_salt_pepper.onnx",
          "soft MoE": d / "task3_soft_moe.onnx"}
sess = {k: ort.InferenceSession(str(p), providers=["CPUExecutionProvider"]) for k, p in models.items() if Path(p).exists()}


def load(name):
    im = Image.open(Path(a.images) / f"{name}.jpg").convert("RGB").resize((128, 128), Image.Resampling.BICUBIC)
    return np.asarray(im, np.float32).transpose(2, 0, 1) / 255


colour = lambda x: float((x.max(0) - x.min(0)).mean())
res = {k: [] for k in ("clean", "corrupted", *sess)}
for e in entries:
    clean = load(e["image"])
    x = apply_spec(clean, e).astype(np.float32)[None]
    res["clean"].append(colour(clean)); res["corrupted"].append(colour(x[0]))
    for k, s in sess.items():
        res[k].append(colour(np.clip(s.run(None, {"image": x})[0][0], 0, 1)))
base = np.mean(res["clean"])
print(f"colourfulness on {len(entries)} salt-and-pepper test images (onnx_dir={d})")
for k, v in res.items():
    print(f"  {k:10s} {np.mean(v):.4f}  ({np.mean(v) / base * 100:.0f}% of the clean image's colour)")
