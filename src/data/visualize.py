"""Visual sanity grids.
  python -m src.data.visualize --out figures/data_sanity          # real test images (needs PETS_IMAGE_DIR / Kaggle)
  python -m src.data.visualize --synthetic --out figures/data_sanity   # no dataset needed
Saves <out>_fixed.png (10 test conditions) and <out>_random.png (random training draws).
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.corruptions import TYPES, apply_spec, make_spec
from src.data.make_manifests import TEST_CONDITIONS


def _synthetic(n):
    g = torch.Generator().manual_seed(0)
    yy, xx = torch.meshgrid(torch.linspace(0, 1, 128), torch.linspace(0, 1, 128), indexing="ij")
    return [((torch.stack([xx, yy, (xx + yy) / 2]) * (0.5 + 0.5 * torch.rand(3, 1, 1, generator=g)))
             + 0.1 * torch.sin(20 * xx * (i + 1))).clamp(0, 1) for i in range(n)]


def _title(s):
    return s["type"] if s["type"] == "clean" else f"{s['type']}\n{s['severity']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/data_sanity")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--rows", type=int, default=3)
    a = ap.parse_args()
    if a.synthetic:
        imgs = _synthetic(a.rows)
    else:
        from src.data.pets import get_images
        imgs = [x.float() / 255 for x in get_images("test")[: a.rows]]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(a.rows, 10, figsize=(20, 2.2 * a.rows), squeeze=False)
    for j, (c, s) in enumerate(TEST_CONDITIONS):
        for i in range(a.rows):
            spec = make_spec(c, rng, s)
            ax[i][j].imshow(apply_spec(imgs[i], spec).permute(1, 2, 0).numpy())
            ax[i][j].axis("off")
            if i == 0:
                t = _title(spec)
                if c == "occlusion":
                    t += f"\ncov={spec['coverage']:.2f}"
                ax[i][j].set_title(t, fontsize=8)
    plt.tight_layout(); plt.savefig(f"{a.out}_fixed.png", dpi=110); plt.close()

    fig, ax = plt.subplots(a.rows, 8, figsize=(16, 2.2 * a.rows), squeeze=False)
    for j in range(8):
        for i in range(a.rows):
            spec = make_spec(TYPES[(j % 4)], rng, "random")
            ax[i][j].imshow(apply_spec(imgs[i], spec).permute(1, 2, 0).numpy()); ax[i][j].axis("off")
            if i == 0:
                ax[i][j].set_title(spec["type"], fontsize=8)
    plt.tight_layout(); plt.savefig(f"{a.out}_random.png", dpi=110); plt.close()
    print("saved", a.out + "_fixed.png", a.out + "_random.png")


if __name__ == "__main__":
    main()
