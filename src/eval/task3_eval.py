"""Task 3 evaluation on the test manifest: reconstruction quality + gate/routing analysis.

  python -m src.eval.task3_eval --ckpt checkpoints/task3_moe.pt --out_dir reports/task3

Outputs:
  metrics_per_type_severity.csv/.json  - L1/SSIM/PSNR per (type,severity), like Task 1
  gate_weights_per_condition.csv       - mean weight per branch for every (type,severity) condition
  routing_heatmap.png                  - conditions x branches heatmap of mean gate weight
  dominant_vs_distributed.png          - 4 lowest-entropy (dominant) + 4 highest-entropy (distributed) examples,
                                         each with clean/corrupted/output + a weight bar chart
  expert_usage_summary.json            - overall mean weight per branch, "home-condition" weight per specialist
                                         (e.g. mean salt_pepper-branch weight when true type is salt_pepper),
                                         and a flag for dead (near-0 everywhere) or dominant (near-1 always) branches
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

from src.data.corruptions import TYPES
from src.data.make_manifests import SEV_NAMES
from src.data.pets import PetsManifestDataset, get_images, load_manifest
from src.models.moe import SoftMoE, build_moe_from_checkpoints
from src.utils import paths
from src.utils.common import get_device, psnr, seed_everything

CONDITION_ORDER = ["clean/none"] + [f"{t}/{s}" for t in TYPES[1:] for s in SEV_NAMES]


def load_moe(ckpt_path, device) -> SoftMoE:
    ck = torch.load(ckpt_path, map_location=device)
    from src.models.autoencoder import build_ae
    from src.models.classifier import build_classifier
    gate = build_classifier(ck["gate_params"])
    experts = {t: build_ae(ck["expert_params"][t]) for t in ck["expert_params"]}
    model = SoftMoE(gate, experts, temperature=ck["params"]["temperature"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


@torch.no_grad()
def run_inference(model, entries, images, device, batch_size=64):
    ds = PetsManifestDataset(entries, images)
    rows, cache = [], {}
    for start in range(0, len(ds), batch_size):
        batch_entries = entries[start:start + batch_size]
        noisy = torch.stack([ds[i][0] for i in range(start, min(start + batch_size, len(ds)))]).to(device)
        clean = torch.stack([ds[i][1] for i in range(start, min(start + batch_size, len(ds)))]).to(device)
        out, weights, _ = model(noisy, return_weights=True)
        out = out.clamp(0, 1)
        l1 = (out - clean).abs().flatten(1).mean(1).cpu().numpy()
        s = ssim(out, clean, data_range=1.0, size_average=False).cpu().numpy()
        p = psnr(out, clean).cpu().numpy()
        w = weights.cpu().numpy()
        for j, e in enumerate(batch_entries):
            rows.append({"idx": e["idx"], "type": e["type"], "severity": e["severity"], "l1": float(l1[j]),
                        "ssim": float(s[j]), "psnr": float(p[j]),
                        **{f"w_{TYPES[k]}": float(w[j, k]) for k in range(4)}})
            cache[e["idx"]] = (clean[j].cpu(), noisy[j].cpu(), out[j].cpu(), w[j])
    return pd.DataFrame(rows), cache


def condition_col(df):
    return np.where(df.type == "clean", "clean/none", df.type + "/" + df.severity)


def routing_heatmap(df, out_path):
    df = df.copy()
    df["condition"] = condition_col(df)
    wcols = [f"w_{t}" for t in TYPES]
    tbl = df.groupby("condition")[wcols].mean().reindex([c for c in CONDITION_ORDER if c in df.condition.unique()])
    fig, ax = plt.subplots(figsize=(6, 0.5 * len(tbl) + 1.5))
    im = ax.imshow(tbl.values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(4)); ax.set_xticklabels(TYPES, rotation=30, ha="right")
    ax.set_yticks(range(len(tbl))); ax.set_yticklabels(tbl.index)
    for i in range(len(tbl)):
        for j in range(4):
            ax.text(j, i, f"{tbl.values[i, j]:.2f}", ha="center", va="center",
                    color="white" if tbl.values[i, j] < 0.5 else "black", fontsize=8)
    ax.set_title("Mean gate weight per condition"); plt.colorbar(im, fraction=0.04)
    plt.tight_layout(); plt.savefig(out_path, dpi=130); plt.close()
    return tbl


def expert_usage_summary(df, tbl):
    wcols = [f"w_{t}" for t in TYPES]
    overall = df[wcols].mean().to_dict()
    home = {}  # for each specialist branch, its weight when the TRUE type matches (higher is better)
    for t in TYPES[1:]:
        sub = df[df.type == t]
        home[t] = float(sub[f"w_{t}"].mean()) if len(sub) else None
    dead = {t: bool(v < 0.05) for t, v in overall.items()}
    dominant_everywhere = {t: bool(v > 0.85) for t, v in overall.items()}
    return {"overall_mean_weight": overall, "home_condition_weight": home,
           "dead_branches": [k.replace("w_", "") for k, v in dead.items() if v],
           "always_dominant_branches": [k.replace("w_", "") for k, v in dominant_everywhere.items() if v],
           "condition_table": tbl.reset_index().to_dict("records")}


def save_dominant_distributed(df, cache, out_path, n=4):
    wcols = [f"w_{t}" for t in TYPES]
    w = df[wcols].values
    entropy = -(w * np.log(np.clip(w, 1e-8, 1))).sum(1)
    df = df.assign(entropy=entropy)
    dominant = df.sort_values("entropy").head(n)
    distributed = df.sort_values("entropy", ascending=False).head(n)
    rows = list(dominant.itertuples()) + list(distributed.itertuples())
    fig, ax = plt.subplots(len(rows), 4, figsize=(11, 2.3 * len(rows)))
    for i, r in enumerate(rows):
        clean, noisy, out, wv = cache[r.idx]
        kind = "dominant" if i < n else "distributed"
        for a, im, name in zip(ax[i][:3], [clean, noisy, out], ["clean", "corrupted", "output"]):
            a.imshow(im.permute(1, 2, 0).numpy().clip(0, 1)); a.axis("off")
            if i in (0, n):
                a.set_title(name, fontsize=8)
        ax[i][3].barh(TYPES, wv); ax[i][3].set_xlim(0, 1)
        ax[i][3].set_title(f"{kind}: {r.type}/{r.severity}\nH={r.entropy:.2f}", fontsize=7)
    plt.tight_layout(); plt.savefig(out_path, dpi=110, bbox_inches="tight"); plt.close()
    print("saved", out_path)


def aggregate_recon(df):
    per_ts = df.groupby(["type", "severity"])[["l1", "ssim", "psnr"]].agg(["mean", "std"])
    per_ts.columns = ["_".join(c) for c in per_ts.columns]
    return per_ts.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(paths.checkpoint_dir() / "task3_moe.pt"))
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--out_dir", default="reports/task3")
    ap.add_argument("--max_items", type=int, default=None)
    a = ap.parse_args()
    seed_everything(42)
    device = get_device()
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    model, ck = load_moe(a.ckpt, device)
    print("loaded task3 checkpoint, temperature =", ck["params"]["temperature"])
    manifest_path = a.manifest or (paths.manifest_dir() / "test_manifest.json")
    _, entries = load_manifest(manifest_path)
    if a.max_items:
        entries = entries[: a.max_items]
    images = get_images("test")

    df, cache = run_inference(model, entries, images, device)
    df.to_csv(out_dir / "per_image_metrics.csv", index=False)

    per_ts = aggregate_recon(df)
    per_ts.to_csv(out_dir / "metrics_per_type_severity.csv", index=False)
    print(per_ts.to_string(index=False))

    tbl = routing_heatmap(df, out_dir / "routing_heatmap.png")
    tbl.to_csv(out_dir / "gate_weights_per_condition.csv")
    summary = expert_usage_summary(df, tbl)
    json.dump(summary, open(out_dir / "expert_usage_summary.json", "w"), indent=2)
    print("overall mean weights:", summary["overall_mean_weight"])
    print("home-condition weights:", summary["home_condition_weight"])
    if summary["dead_branches"]:
        print("WARNING dead branches (mean weight < 0.05 everywhere):", summary["dead_branches"])

    save_dominant_distributed(df, cache, out_dir / "dominant_vs_distributed.png")


if __name__ == "__main__":
    main()
