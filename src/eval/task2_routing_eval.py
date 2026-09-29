"""Hard-routed restoration evaluation on the test manifest: oracle routing (true label) vs predicted
routing (classifier's argmax), quantifying how much classifier error costs in restoration quality.

  python -m src.eval.task2_routing_eval \
      --classifier checkpoints/task2_classifier.pt \
      --salt_pepper checkpoints/task2_specialist_salt_pepper.pt \
      --blur checkpoints/task2_specialist_blur.pt \
      --occlusion checkpoints/task2_specialist_occlusion.pt \
      --out_dir reports/task2_routing

Outputs:
  routing_metrics.csv / .json   - per-image L1/SSIM/PSNR for BOTH oracle and predicted routing, plus routing labels
  routing_summary.json          - overall + per-type aggregates for oracle vs predicted, mean inference time
  misroute_examples.png         - examples where predicted route != oracle route, with probs/expert/timing shown
                                  (mirrors the "Hard-Routed Restoration" app workspace: 4 probs, predicted class,
                                  chosen expert, output, time)
  routing_confusion_delta.json  - discussion aid: for each (true_type -> predicted_type) pair, mean SSIM drop
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pytorch_msssim import ssim

from src.data.corruptions import TYPES
from src.data.pets import PetsManifestDataset, get_images, load_manifest
from src.models.routing import load_classifier, load_specialists, restore_batch
from src.utils import paths
from src.utils.common import get_device, psnr, seed_everything


@torch.no_grad()
def run(classifier, specialists, entries, images, device, batch_size=32):
    ds = PetsManifestDataset(entries, images)
    rows, cache = [], {}
    cls_time_total, n_total = 0.0, 0
    for start in range(0, len(ds), batch_size):
        batch = entries[start:start + batch_size]
        noisy = torch.stack([ds[i][0] for i in range(start, min(start + batch_size, len(ds)))])
        clean = torch.stack([ds[i][1] for i in range(start, min(start + batch_size, len(ds)))])
        label = torch.tensor([e["label"] for e in batch])

        t0 = time.perf_counter()
        probs = torch.softmax(classifier(noisy.to(device)), dim=1).cpu()
        cls_time_total += time.perf_counter() - t0
        n_total += len(batch)
        pred = probs.argmax(1)

        t0 = time.perf_counter()
        out_pred = restore_batch(noisy, pred, specialists, device)
        t_pred_total = time.perf_counter() - t0
        t0 = time.perf_counter()
        out_oracle = restore_batch(noisy, label, specialists, device)
        t_oracle_total = time.perf_counter() - t0

        l1_p = (out_pred - clean).abs().flatten(1).mean(1).numpy()
        l1_o = (out_oracle - clean).abs().flatten(1).mean(1).numpy()
        s_p = ssim(out_pred, clean, data_range=1.0, size_average=False).numpy()
        s_o = ssim(out_oracle, clean, data_range=1.0, size_average=False).numpy()
        p_p = psnr(out_pred, clean).numpy()
        p_o = psnr(out_oracle, clean).numpy()

        for j, e in enumerate(batch):
            rows.append({"idx": e["idx"], "type": e["type"], "severity": e["severity"], "true_label": int(label[j]),
                        "pred_label": int(pred[j]), "true_type": TYPES[int(label[j])],
                        "pred_type": TYPES[int(pred[j])], "misrouted": bool(pred[j] != label[j]),
                        "l1_pred": float(l1_p[j]), "ssim_pred": float(s_p[j]), "psnr_pred": float(p_p[j]),
                        "l1_oracle": float(l1_o[j]), "ssim_oracle": float(s_o[j]), "psnr_oracle": float(p_o[j]),
                        "probs": probs[j].tolist(),
                        "time_ms_pred": 1000 * (t_pred_total / len(batch)), "time_ms_oracle": 1000 * (t_oracle_total / len(batch))})
            cache[e["idx"]] = (clean[j], noisy[j], out_pred[j], out_oracle[j], probs[j])
    return pd.DataFrame(rows), cache, 1000 * cls_time_total / max(n_total, 1)


def summarize(df):
    def agg(prefix):
        overall = {"l1": df[f"l1_{prefix}"].mean(), "ssim": df[f"ssim_{prefix}"].mean(), "psnr": df[f"psnr_{prefix}"].mean()}
        per_type = df.groupby("type")[[f"l1_{prefix}", f"ssim_{prefix}", f"psnr_{prefix}"]].mean()
        per_type.columns = ["l1", "ssim", "psnr"]
        return overall, per_type.reset_index().to_dict("records")
    oracle_overall, oracle_per_type = agg("oracle")
    pred_overall, pred_per_type = agg("pred")
    return {"oracle": {"overall": oracle_overall, "per_type": oracle_per_type},
           "predicted": {"overall": pred_overall, "per_type": pred_per_type},
           "misroute_rate": float(df.misrouted.mean()),
           "n": len(df)}


def confusion_delta(df):
    """For every (true_type -> predicted_type) pair actually observed: mean SSIM/L1 cost of the misroute
    relative to oracle routing. Helps discuss classifier-caused failures concretely."""
    out = []
    for (tt, pt), g in df.groupby(["true_type", "pred_type"]):
        out.append({"true_type": tt, "pred_type": pt, "n": len(g),
                    "mean_ssim_drop": float((g.ssim_oracle - g.ssim_pred).mean()),
                    "mean_l1_increase": float((g.l1_pred - g.l1_oracle).mean())})
    return sorted(out, key=lambda r: -r["mean_ssim_drop"])


def save_misroute_examples(df, cache, out_path, n=8, seed=0):
    mis = df[df.misrouted]
    if len(mis) == 0:
        print("no misrouted examples in this manifest -- classifier was perfect here"); return
    worst = mis.sort_values("ssim_pred").head(n)  # worst-quality misroutes first
    fig, ax = plt.subplots(len(worst), 5, figsize=(14, 2.6 * len(worst)))
    if len(worst) == 1:
        ax = ax[None, :]
    for i, (_, r) in enumerate(worst.iterrows()):
        clean, noisy, out_pred, out_oracle, probs = cache[r["idx"]]
        for a, im, name in zip(ax[i][:4], [clean, noisy, out_pred, out_oracle],
                               ["clean", "corrupted", "output (predicted route)", "output (oracle route)"]):
            a.imshow(im.permute(1, 2, 0).numpy().clip(0, 1)); a.axis("off")
            if i == 0:
                a.set_title(name, fontsize=8)
        ax[i][4].barh(TYPES, probs.numpy()); ax[i][4].set_xlim(0, 1)
        ax[i][4].set_title(f"true={r['true_type']} pred={r['pred_type']}\n"
                            f"ssim {r['ssim_oracle']:.2f}->{r['ssim_pred']:.2f}  {r['time_ms_pred']:.1f}ms", fontsize=7)
    plt.tight_layout(); plt.savefig(out_path, dpi=110, bbox_inches="tight"); plt.close()
    print("saved", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classifier", default=str(paths.checkpoint_dir() / "task2_classifier.pt"))
    ap.add_argument("--salt_pepper", default=str(paths.checkpoint_dir() / "task2_specialist_salt_pepper.pt"))
    ap.add_argument("--blur", default=str(paths.checkpoint_dir() / "task2_specialist_blur.pt"))
    ap.add_argument("--occlusion", default=str(paths.checkpoint_dir() / "task2_specialist_occlusion.pt"))
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--out_dir", default="reports/task2_routing")
    ap.add_argument("--max_items", type=int, default=None)
    a = ap.parse_args()
    seed_everything(42)
    device = get_device()
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    classifier, _ = load_classifier(a.classifier, device)
    specialists = load_specialists({"salt_pepper": a.salt_pepper, "blur": a.blur, "occlusion": a.occlusion}, device)

    manifest_path = a.manifest or (paths.manifest_dir() / "test_manifest.json")
    _, entries = load_manifest(manifest_path)
    if a.max_items:
        entries = entries[: a.max_items]
    images = get_images("test")

    df, cache, cls_ms = run(classifier, specialists, entries, images, device)
    df.drop(columns="probs").to_csv(out_dir / "routing_metrics.csv", index=False)
    df.to_json(out_dir / "routing_metrics.json", orient="records", indent=2)

    summary = summarize(df)
    summary["mean_classifier_ms_per_image"] = cls_ms
    json.dump(summary, open(out_dir / "routing_summary.json", "w"), indent=2)
    json.dump(confusion_delta(df), open(out_dir / "routing_confusion_delta.json", "w"), indent=2)

    print(f"misroute rate: {summary['misroute_rate']:.3%}  (n={summary['n']})")
    print("ORACLE   ", summary["oracle"]["overall"])
    print("PREDICTED", summary["predicted"]["overall"])
    print(f"mean classifier latency: {cls_ms:.2f} ms/image")

    save_misroute_examples(df, cache, out_dir / "misroute_examples.png")


if __name__ == "__main__":
    main()
