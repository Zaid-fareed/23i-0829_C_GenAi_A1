"""Task 1 evaluation on the deterministic TEST manifest.

  python -m src.eval.task1_eval --ckpt checkpoints/task1_dae.pt --out_dir reports/task1

Outputs (all under --out_dir):
  metrics_per_type_severity.csv / .json   - L1/SSIM/PSNR aggregated per (corruption type, severity)
  metrics_overall.json                    - overall + per-type-only aggregates
  examples_grid.png                       - >=12 examples: clean | corrupted | output | abs-error, across all
                                             corruption types and severities
  failure_cases.png                       - the 4 worst examples by SSIM (with their type/severity labelled)
  per_image_metrics.csv                   - every test item's L1/SSIM/PSNR/type/severity (for the report/appendix)
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

from src.data.pets import PetsManifestDataset, get_images, load_manifest
from src.models.autoencoder import build_ae
from src.utils import paths
from src.utils.common import get_device, psnr, seed_everything

SEV_ORDER = ["none", "low", "medium", "high"]


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device)
    model = build_ae(ck["params"]).to(device).eval()
    model.load_state_dict(ck["state_dict"])
    return model, ck


@torch.no_grad()
def run_inference(model, entries, images, device, batch_size=64):
    """Returns a DataFrame with one row per entry: idx,type,severity,label,l1,ssim,psnr,
    plus in-memory arrays of (clean, noisy, output) for later plotting."""
    ds = PetsManifestDataset(entries, images)
    rows, cache = [], {}
    for start in range(0, len(ds), batch_size):
        batch_entries = entries[start:start + batch_size]
        noisy = torch.stack([ds[i][0] for i in range(start, min(start + batch_size, len(ds)))]).to(device)
        clean = torch.stack([ds[i][1] for i in range(start, min(start + batch_size, len(ds)))]).to(device)
        out = model(noisy).clamp(0, 1)
        l1 = (out - clean).abs().flatten(1).mean(1).cpu().numpy()
        s = ssim(out, clean, data_range=1.0, size_average=False).cpu().numpy()
        p = psnr(out, clean).cpu().numpy()
        for j, e in enumerate(batch_entries):
            rows.append({"idx": e["idx"], "type": e["type"], "severity": e["severity"],
                        "l1": float(l1[j]), "ssim": float(s[j]), "psnr": float(p[j])})
            cache[e["idx"]] = (clean[j].cpu(), noisy[j].cpu(), out[j].cpu())
    return pd.DataFrame(rows), cache


def aggregate(df: pd.DataFrame):
    per_ts = df.groupby(["type", "severity"])[["l1", "ssim", "psnr"]].agg(["mean", "std", "count"])
    per_ts.columns = ["_".join(c) for c in per_ts.columns]
    per_ts = per_ts.reset_index()
    per_ts["_sev_order"] = per_ts["severity"].map({s: i for i, s in enumerate(SEV_ORDER)})
    per_ts = per_ts.sort_values(["type", "_sev_order"]).drop(columns="_sev_order")
    per_type = df.groupby("type")[["l1", "ssim", "psnr"]].mean().reset_index()
    overall = {"n": len(df), "l1": df.l1.mean(), "ssim": df.ssim.mean(), "psnr": df.psnr.mean()}
    return per_ts, per_type, overall


def _panel(ax_row, clean, noisy, out, title):
    err = (out - clean).abs().mean(0)
    imgs = [clean.permute(1, 2, 0), noisy.permute(1, 2, 0), out.permute(1, 2, 0)]
    for ax, im, name in zip(ax_row[:3], imgs, ["clean", "corrupted", "output"]):
        ax.imshow(im.numpy().clip(0, 1)); ax.axis("off")
        if title:
            ax.set_title(name, fontsize=8)
    ax_row[3].imshow(err.numpy(), cmap="inferno", vmin=0, vmax=0.5); ax_row[3].axis("off")
    if title:
        ax_row[3].set_title("abs error", fontsize=8)
    ax_row[0].text(-0.15, 0.5, title, transform=ax_row[0].transAxes, ha="right", va="center", fontsize=8, rotation=0)


def save_examples_grid(df, cache, out_path, n_per_condition=1, seed=0):
    """>=12 examples spanning every (type,severity) condition present in the test manifest."""
    rng = np.random.default_rng(seed)
    conditions = df[["type", "severity"]].drop_duplicates().sort_values(["type", "severity"]).values.tolist()
    rows = []
    for t, s in conditions:
        sub = df[(df.type == t) & (df.severity == s)]
        rows.append(sub.sample(1, random_state=int(rng.integers(1e6))).iloc[0])
    fig, ax = plt.subplots(len(rows), 4, figsize=(8, 2 * len(rows)))
    if len(rows) == 1:
        ax = ax[None, :]
    for i, r in enumerate(rows):
        clean, noisy, out = cache[r["idx"]]
        _panel(ax[i], clean, noisy, out, f"{r['type']}\n{r['severity']}" if r["type"] != "clean" else "clean")
    plt.tight_layout(); plt.savefig(out_path, dpi=110, bbox_inches="tight"); plt.close()
    print(f"saved {out_path} ({len(rows)} examples)")


def save_failure_cases(df, cache, out_path, n=4):
    worst = df.sort_values("ssim").head(n)
    fig, ax = plt.subplots(n, 4, figsize=(8, 2 * n))
    for i, (_, r) in enumerate(worst.iterrows()):
        clean, noisy, out = cache[r["idx"]]
        label = f"{r['type']}/{r['severity']}\nssim={r['ssim']:.2f} l1={r['l1']:.3f}"
        _panel(ax[i], clean, noisy, out, label)
    plt.tight_layout(); plt.savefig(out_path, dpi=110, bbox_inches="tight"); plt.close()
    print(f"saved {out_path}")
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(paths.checkpoint_dir() / "task1_dae.pt"))
    ap.add_argument("--manifest", default=None, help="default: manifests/test_manifest.json")
    ap.add_argument("--out_dir", default="reports/task1")
    ap.add_argument("--max_items", type=int, default=None)
    a = ap.parse_args()
    seed_everything(42)
    device = get_device()
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    model, ck = load_model(a.ckpt, device)
    print("loaded checkpoint from epoch", ck.get("epoch"), "params", ck.get("params"))
    manifest_path = a.manifest or (paths.manifest_dir() / "test_manifest.json")
    meta, entries = load_manifest(manifest_path)
    if a.max_items:
        entries = entries[: a.max_items]
    images = get_images("test")

    df, cache = run_inference(model, entries, images, device)
    df.to_csv(out_dir / "per_image_metrics.csv", index=False)

    per_ts, per_type, overall = aggregate(df)
    per_ts.to_csv(out_dir / "metrics_per_type_severity.csv", index=False)
    json.dump({"per_type_severity": per_ts.to_dict("records"), "per_type": per_type.to_dict("records"),
              "overall": overall, "manifest_meta": meta, "ckpt_params": ck.get("params")},
             open(out_dir / "metrics_overall.json", "w"), indent=2)
    print(per_ts.to_string(index=False))
    print("OVERALL", overall)

    save_examples_grid(df, cache, out_dir / "examples_grid.png")
    worst = save_failure_cases(df, cache, out_dir / "failure_cases.png")
    print("failure cases:\n", worst[["idx", "type", "severity", "l1", "ssim", "psnr"]].to_string(index=False))


if __name__ == "__main__":
    main()
