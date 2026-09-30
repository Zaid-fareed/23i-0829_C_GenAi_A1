"""Independent re-measurement of the reported TEST results using ONLY the deployed artefacts:
  ONNX models + backend numpy corruption code + scikit-image SSIM (a different SSIM implementation than training).
Compares the numbers with what the Kaggle evaluation scripts reported (reports/*). No PyTorch is used.

  PYTHONPATH=backend python scripts/verify_onnx_results.py --images data/oxford-iiit-pet/images [--n 1200]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
from PIL import Image
from skimage.metrics import structural_similarity

from app.corruptions import TYPES, apply_spec

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--images", default=str(ROOT / "data/oxford-iiit-pet/images"))
ap.add_argument("--n", type=int, default=0, help="use only the first N test entries (0 = all)")
ap.add_argument("--t3_reports", default="reports/task3", help="folder with the Kaggle Task 3 per_image_metrics.csv to compare against")
ap.add_argument("--routing_reports", default="reports/task2_routing", help="folder with the Kaggle routing_summary.json to compare against")
a = ap.parse_args()

entries = json.load(open(ROOT / "manifests/test_manifest.json"))["entries"]
if a.n:
    entries = entries[:a.n]
sess = {k: ort.InferenceSession(str(ROOT / "onnx_models" / f), providers=["CPUExecutionProvider"]) for k, f in {
    "t1": "task1_universal_dae.onnx", "clf": "task2_classifier.onnx", "sp": "task2_specialist_salt_pepper.onnx",
    "bl": "task2_specialist_blur.onnx", "oc": "task2_specialist_occlusion.onnx", "t3": "task3_soft_moe.onnx"}.items()}
EXPERT = {1: "sp", 2: "bl", 3: "oc"}


def load(name):
    im = Image.open(Path(a.images) / f"{name}.jpg").convert("RGB").resize((128, 128), Image.Resampling.BICUBIC)
    return np.ascontiguousarray(np.asarray(im, np.float32).transpose(2, 0, 1) / 255.0)


def metrics(out, clean):
    out = np.clip(out, 0, 1)
    l1 = float(np.abs(out - clean).mean())
    psnr = float(10 * np.log10(1.0 / max(float(((out - clean) ** 2).mean()), 1e-10)))
    ssim = float(structural_similarity(out.transpose(1, 2, 0), clean.transpose(1, 2, 0), channel_axis=2, data_range=1.0,
                                       gaussian_weights=True, sigma=1.5, use_sample_covariance=False))
    return l1, psnr, ssim


rows = []
for i, e in enumerate(entries):
    clean = load(e["image"])
    x = apply_spec(clean, e).astype(np.float32)[None]
    t1 = sess["t1"].run(None, {"image": x})[0][0]
    logits = sess["clf"].run(None, {"image": x})[0][0]
    pred = int(logits.argmax())
    hard = x[0] if pred == 0 else sess[EXPERT[pred]].run(None, {"image": x})[0][0]
    t3 = sess["t3"].run(None, {"image": x})[0][0]
    r = {"type": e["type"], "severity": e["severity"], "label": e["label"], "pred": pred}
    for tag, out in (("t1", t1), ("hard", hard), ("soft", t3), ("input", x[0])):
        r[f"{tag}_l1"], r[f"{tag}_psnr"], r[f"{tag}_ssim"] = metrics(out, clean)
    rows.append(r)
    if (i + 1) % 300 == 0:
        print(f"  processed {i + 1}/{len(entries)}", flush=True)
df = pd.DataFrame(rows)
n = len(df)
print(f"\n=== independent re-measurement on {n} TEST entries (ONNX + numpy corruptions + skimage SSIM) ===")
print(f"classifier accuracy: {(df.label == df.pred).mean():.4f}   (Kaggle report: 0.9910)")
print("\n                       SSIM(remeasured)  SSIM(Kaggle report)")
rep1 = pd.read_csv(ROOT / "reports/task1/per_image_metrics.csv")
rep3 = pd.read_csv(ROOT / a.t3_reports / "per_image_metrics.csv")
if n == len(rep1):
    print(f"Task1 universal AE   :   {df.t1_ssim.mean():.4f}            {rep1.ssim.mean():.4f}")
    print(f"Task3 soft MoE       :   {df.soft_ssim.mean():.4f}            {rep3.ssim.mean():.4f}")
else:
    idx = rep1.iloc[:n]
    print(f"Task1 universal AE   :   {df.t1_ssim.mean():.4f}            {idx.ssim.mean():.4f}   (same first {n} entries)")
    print(f"Task3 soft MoE       :   {df.soft_ssim.mean():.4f}            {rep3.iloc[:n].ssim.mean():.4f}")
rs = json.load(open(ROOT / a.routing_reports / "routing_summary.json"))["predicted"]["overall"]["ssim"]
print(f"Task2 hard (predicted):  {df.hard_ssim.mean():.4f}            {rs:.4f}" + ("" if n == len(rep1) else "  (Kaggle = full set)"))
print(f"do-nothing baseline  :   {df.input_ssim.mean():.4f}")
print("\nper-type SSIM (remeasured): input -> universal / hard / soft")
print(df.groupby("type")[["input_ssim", "t1_ssim", "hard_ssim", "soft_ssim"]].mean().round(4).to_string())
