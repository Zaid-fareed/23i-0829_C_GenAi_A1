"""Task 4 evaluation on the official FS2K TEST set (run once, after training/Optuna are finished).

  python -m src.eval.task4_eval --ckpt checkpoints/task4_generator.pt --out_dir reports/task4_eval

Outputs: metrics_per_style.csv/json (L1/SSIM/PSNR overall and per style), examples_grid.png
(photo | ground truth | generated | abs-error, 12 examples), styles_grid.png (one photo rendered in all 3 styles),
failure_cases.png (4 worst SSIM).
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pytorch_msssim import ssim

from src.data.fs2k import build_loader
from src.models.cgan import UNetGenerator
from src.utils import paths
from src.utils.common import get_device, psnr


def _rows(items, path, title):
    n = len(items)
    fig, ax = plt.subplots(n, 4, figsize=(7, 1.8 * n))
    ax = np.atleast_2d(ax)
    for r, (x, y, o, lab) in enumerate(items):
        err = (o - y).abs()
        for c, (im, name) in enumerate(((x, "photo"), (y, "ground truth"), (o, "generated"), (err, "abs error"))):
            a = ax[r, c]
            a.imshow(im.permute(1, 2, 0).squeeze(-1) if im.size(0) == 1 else im.permute(1, 2, 0),
                     cmap="gray" if im.size(0) == 1 else None, vmin=0, vmax=1 if c < 3 else 0.5)
            a.axis("off")
            if r == 0:
                a.set_title(name, fontsize=8)
        ax[r, 0].text(-0.05, 0.5, lab, transform=ax[r, 0].transAxes, ha="right", va="center", fontsize=6)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(paths.checkpoint_dir() / "task4_generator.pt"))
    ap.add_argument("--out_dir", default="reports/task4_eval")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = get_device()
    ck = torch.load(a.ckpt, map_location=device)
    p = ck["params"]
    G = UNetGenerator(p["base"], p["emb_dim"], p["dropout"]).to(device).eval()
    G.load_state_dict(ck["state_dict"])

    X, Y, O, S = [], [], [], []
    with torch.no_grad():
        for x, y, s in build_loader("test", 32, shuffle=False, workers=0):
            o = G(x.to(device), s.to(device)).cpu()
            X.append((x + 1) / 2); Y.append((y + 1) / 2); O.append((o + 1) / 2); S.append(s)
    X, Y, O, S = torch.cat(X), torch.cat(Y), torch.cat(O), torch.cat(S)
    df = pd.DataFrame({"style": S.numpy() + 1, "l1": (O - Y).abs().flatten(1).mean(1).numpy(),
                       "ssim": ssim(O, Y, data_range=1.0, size_average=False).numpy(), "psnr": psnr(O, Y).numpy()})
    per = df.groupby("style")[["l1", "ssim", "psnr"]].agg(["mean", "std", "count"])
    per.columns = ["_".join(c) for c in per.columns]
    per.to_csv(out / "metrics_per_style.csv")
    res = {"n": len(df), **{k: float(df[k].mean()) for k in ("l1", "ssim", "psnr")}, "per_style": per.to_dict("index")}
    json.dump(res, open(out / "metrics_per_style.json", "w"), indent=2, default=float)
    print(per.round(4))
    print("overall", {k: round(res[k], 4) for k in ("l1", "ssim", "psnr")})

    idx = np.linspace(0, len(df) - 1, 12).astype(int)
    _rows([(X[i], Y[i], O[i], f"style {S[i] + 1}") for i in idx], out / "examples_grid.png", "Test examples")
    worst = np.argsort(df["ssim"].values)[:4]
    _rows([(X[i], Y[i], O[i], f"style {S[i] + 1}\nSSIM {df.ssim[i]:.2f}") for i in worst],
          out / "failure_cases.png", "Failure cases (lowest SSIM)")

    fig, ax = plt.subplots(4, 4, figsize=(6.4, 6.4))
    with torch.no_grad():
        for r, i in enumerate(idx[:4]):
            ax[r, 0].imshow(X[i].permute(1, 2, 0)); ax[r, 0].axis("off")
            for s in range(3):
                o = (G(((X[i:i + 1] * 2 - 1)).to(device), torch.tensor([s], device=device)).cpu() + 1) / 2
                ax[r, s + 1].imshow(o[0, 0], cmap="gray", vmin=0, vmax=1); ax[r, s + 1].axis("off")
    for c, t in enumerate(["photo", "Style 1", "Style 2", "Style 3"]):
        ax[0, c].set_title(t, fontsize=8)
    fig.tight_layout(); fig.savefig(out / "styles_grid.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
