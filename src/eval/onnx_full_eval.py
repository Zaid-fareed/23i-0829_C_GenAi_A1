"""Final-protocol test evaluation of the DEPLOYED ONNX models on the FULL test manifest.

The full manifest gives every clean test image all 10 conditions (clean + {salt-and-pepper, blur, occlusion} x
{low, medium, high}, using the fixed severities of the assignment), i.e. 3,669 x 10 = 36,690 deterministic inputs.
For every input this script records, for the universal AE (Task 1), the hard-routed system with predicted and with
oracle routing (Task 2) and the soft mixture-of-experts (Task 3):  L1, SSIM, PSNR, plus the classifier
probabilities and the four gate weights.  Corruptions are applied with the exact training-time code
(src.data.corruptions.apply_spec) from the stored manifest specs.

  python -m src.eval.onnx_full_eval --onnx_dir onnx_models --out_dir reports/final_full_test
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from PIL import Image
from pytorch_msssim import ssim

from src.data.corruptions import TYPES, apply_spec
from src.utils.common import psnr

ROOT = Path(__file__).resolve().parents[2]
MODELS = {"t1": "task1_universal_dae.onnx", "clf": "task2_classifier.onnx",
          "sp": "task2_specialist_salt_pepper.onnx", "bl": "task2_specialist_blur.onnx",
          "oc": "task2_specialist_occlusion.onnx", "moe": "task3_soft_moe.onnx"}
EXPERT = {1: "sp", 2: "bl", 3: "oc"}  # class id -> specialist key (0 = clean -> identity bypass)


def load_sessions(onnx_dir, threads=0):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads or os.cpu_count() or 4
    return {k: ort.InferenceSession(str(Path(onnx_dir) / f), opts, providers=["CPUExecutionProvider"])
            for k, f in MODELS.items()}


def load_image(images_dir, name, size=128):
    im = Image.open(Path(images_dir) / f"{name}.jpg").convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    return np.ascontiguousarray(np.asarray(im, np.float32).transpose(2, 0, 1) / 255.0)


def batch_metrics(out, clean):
    out = out.clamp(0, 1)
    l1 = (out - clean).abs().flatten(1).mean(1)
    return l1.numpy(), ssim(out, clean, data_range=1.0, size_average=False).numpy(), psnr(out, clean).numpy()


def run_expert(sess, key, x, idx):
    """Run specialist `key` on x[idx] (idx: indices into the batch); returns array of outputs."""
    return sess[key].run(None, {"image": x[idx]})[0]


def route(sess, x, classes):
    """Hard routing: classes[i] == 0 -> identity bypass, else the matching specialist."""
    out = x.copy()
    for c, key in EXPERT.items():
        idx = np.where(classes == c)[0]
        if len(idx):
            out[idx] = run_expert(sess, key, x, idx)
    return out


def run_batch(sess, clean, entries):
    """clean: (B,3,H,W) float32, entries: list of manifest dicts. Returns (per-image DataFrame, dict of output arrays)."""
    x = np.stack([apply_spec(torch.from_numpy(c), e).numpy() for c, e in zip(clean, entries)]).astype(np.float32)
    labels = np.array([e["label"] for e in entries])
    t1 = sess["t1"].run(None, {"image": x})[0]
    logits = sess["clf"].run(None, {"image": x})[0]
    e = np.exp(logits - logits.max(1, keepdims=True)); probs = e / e.sum(1, keepdims=True)
    pred = probs.argmax(1)
    hard_pred, hard_oracle = route(sess, x, pred), route(sess, x, labels)
    moe_out, w = sess["moe"].run(None, {"image": x})
    c = torch.from_numpy(clean)
    rows = {"idx": [e["idx"] for e in entries], "image": [e["image"] for e in entries],
            "type": [e["type"] for e in entries], "severity": [e["severity"] for e in entries],
            "label": labels, "pred": pred}
    for i, t in enumerate(TYPES):
        rows[f"p_{t}"] = probs[:, i]
        rows[f"w_{t}"] = w[:, i]
    outs = {"input": x, "t1": t1, "hard_pred": hard_pred, "hard_oracle": hard_oracle, "soft": moe_out}
    for tag, arr in outs.items():
        rows[f"{tag}_l1"], rows[f"{tag}_ssim"], rows[f"{tag}_psnr"] = batch_metrics(torch.from_numpy(np.asarray(arr)), c)
    return pd.DataFrame(rows), outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx_dir", default=str(ROOT / "onnx_models"))
    ap.add_argument("--manifest", default=str(ROOT / "manifests/test_manifest_full.json"))
    ap.add_argument("--images", default=str(ROOT / "data/oxford-iiit-pet/images"))
    ap.add_argument("--out_dir", default=str(ROOT / "reports/final_full_test"))
    ap.add_argument("--n", type=int, default=0, help="only the first N entries (0 = all)")
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--start", type=int, default=0, help="first entry (for splitting the run over processes)")
    ap.add_argument("--stop", type=int, default=0, help="stop before this entry (0 = end)")
    ap.add_argument("--threads", type=int, default=0, help="ONNX Runtime threads per process")
    ap.add_argument("--merge", action="store_true", help="merge per_image_metrics_part*.csv.gz into per_image_metrics.csv.gz and exit")
    a = ap.parse_args()
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    if a.merge:
        parts = sorted(out_dir.glob("per_image_metrics_part*.csv.gz"), key=lambda p: int(p.stem.split("part")[1].split(".")[0]))
        df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True).sort_values("idx").reset_index(drop=True)
        df.to_csv(out_dir / "per_image_metrics.csv.gz", index=False, compression="gzip")
        for p in parts:
            p.unlink()
        print("merged", len(parts), "parts ->", len(df), "rows")
        return
    meta, entries = json.load(open(a.manifest))["meta"], json.load(open(a.manifest))["entries"]
    if a.n:
        entries = entries[:a.n]
    entries = entries[a.start:(a.stop or None)]
    sess = load_sessions(a.onnx_dir, a.threads)
    cache, frames, t0 = {}, [], time.time()
    for s in range(0, len(entries), a.batch):
        chunk = entries[s:s + a.batch]
        for e in chunk:
            if e["image"] not in cache:
                cache[e["image"]] = load_image(a.images, e["image"])
        df, _ = run_batch(sess, np.stack([cache[e["image"]] for e in chunk]), chunk)
        frames.append(df)
        done = s + len(chunk)
        if (done // a.batch) % 40 == 0 or done == len(entries):
            el = time.time() - t0
            print(f"  {done}/{len(entries)}  {el / 60:.1f} min elapsed, ETA {(el / done) * (len(entries) - done) / 60:.1f} min", flush=True)
    df = pd.concat(frames, ignore_index=True)
    part = f"_part{a.start}" if (a.start or a.stop) else ""
    df.to_csv(out_dir / f"per_image_metrics{part}.csv.gz", index=False, compression="gzip")
    json.dump({"manifest_meta": meta, "n_entries": len(df), "onnx_dir": str(a.onnx_dir),
               "models": {k: (Path(a.onnx_dir) / f).name for k, f in MODELS.items()}}, open(out_dir / "run_info.json", "w"), indent=2)
    print("saved", out_dir / f"per_image_metrics{part}.csv.gz", len(df), "rows")


if __name__ == "__main__":
    main()
