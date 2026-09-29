"""Memorization check (Task 1 / any single AE checkpoint): does the model generalize or just memorize its train subset?

  python -m src.eval.generalization --ckpt checkpoints/task1_dae.pt --history checkpoints/task1_history.json \
      --n_train 736 --out_dir reports/task1_generalization

Compares SSIM/PSNR/L1 on
  (a) images the model was TRAINED on (fixed-seed corruptions, same recipe as the val manifest),
  (b) the unseen validation manifest.
A large train-vs-val gap (e.g. train SSIM >> val SSIM) means memorization; a small gap means generalization.
Also plots the train/val loss curves from the saved history.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.make_manifests import build_entries
from src.data.pets import PetsManifestDataset, get_images, load_manifest, load_split
from src.eval.task1_eval import load_model, run_inference
from src.utils import paths
from src.utils.common import get_device


def summarize(df):
    return {"ssim": float(df["ssim"].mean()), "psnr": float(df["psnr"].mean()), "l1": float(df["l1"].mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(paths.checkpoint_dir() / "task1_dae.pt"))
    ap.add_argument("--history", default=None)
    ap.add_argument("--n_train", type=int, default=736, help="how many TRAIN images to evaluate (match val size)")
    ap.add_argument("--out_dir", default="reports/task1_generalization")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = get_device()
    model, _ = load_model(a.ckpt, device)

    names = load_split()["train"][:a.n_train]  # same prefix the --max_train subset uses
    tr_entries = build_entries(names, "val", seed=42)
    tr_df = run_inference(model, tr_entries, get_images("train")[:a.n_train], device)[0]
    _, va_entries = load_manifest(paths.manifest_dir() / "val_manifest.json")
    va_df = run_inference(model, va_entries, get_images("val"), device)[0]

    res = {"train_subset": summarize(tr_df), "val": summarize(va_df)}
    res["gap_ssim"] = res["train_subset"]["ssim"] - res["val"]["ssim"]
    res["gap_psnr"] = res["train_subset"]["psnr"] - res["val"]["psnr"]
    res["per_type"] = {t: {"train_ssim": float(tr_df[tr_df.type == t].ssim.mean()),
                           "val_ssim": float(va_df[va_df.type == t].ssim.mean())} for t in sorted(tr_df.type.unique())}
    json.dump(res, open(out / "generalization.json", "w"), indent=2)
    print(json.dumps(res, indent=2))

    if a.history and Path(a.history).exists():
        h = json.load(open(a.history))
        ep = [r["epoch"] + 1 for r in h]
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
        ax[0].plot(ep, [r["train_loss"] for r in h], label="train loss")
        ax[0].set_title("Training loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
        ax[1].plot(ep, [1 - r["train_ssim"] for r in h], label="train (1-SSIM)")
        ax[1].plot(ep, [1 - r["val_ssim"] for r in h], label="val (1-SSIM)")
        ax[1].set_title("Train vs validation (1-SSIM)"); ax[1].set_xlabel("epoch"); ax[1].legend()
        fig.tight_layout(); fig.savefig(out / "curves.png", dpi=130)


if __name__ == "__main__":
    main()
