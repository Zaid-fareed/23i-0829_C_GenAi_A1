"""Stage 1 of the report build: figures that do NOT need the full-test evaluation.

  python report/build_figures.py            # writes report/figures/*.pdf|png

Architecture diagrams, training curves (from the saved histories), Optuna studies, the corruption grid,
the Task 4 sample progression and the Stitch design montage.  Every plotted number comes from a saved result file.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "legend.fontsize": 6.5,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight"})
C = {"in": "#e2e8f0", "enc": "#c7d2fe", "lat": "#fde68a", "dec": "#bbf7d0", "out": "#e2e8f0", "gate": "#fbcfe8",
     "id": "#f1f5f9", "sp": "#fecaca", "bl": "#bae6fd", "oc": "#bbf7d0", "ed": "#4f46e5"}


def hist(name):
    return pd.DataFrame(json.load(open(ROOT / "checkpoints" / f"{name}.json")))


sys.path.insert(0, str(Path(__file__).resolve().parent))
from arch_diagrams import fig_arch_ae, fig_arch_cgan, fig_arch_routing, fig_arch_system  # noqa: E402


# ------------------------------------------------------------------------------------------ training curves
def fig_curves_task1():
    h = hist("task1_history")
    fig, ax = plt.subplots(1, 3, figsize=(7.1, 1.9))
    ax[0].plot(h.epoch + 1, h.train_loss, label="train $\\mathcal{L}_{UDAE}$ (random corruptions)")
    ax[0].plot(h.epoch + 1, h.val_obj, label="val objective (0.5·L1+0.5·(1−SSIM))")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend(frameon=False)
    ax[1].plot(h.epoch + 1, h.train_ssim, label="train"); ax[1].plot(h.epoch + 1, h.val_ssim, label="validation")
    ax[1].set_title("SSIM (train vs validation)"); ax[1].set_xlabel("epoch"); ax[1].legend(frameon=False)
    for t in ("clean", "salt_pepper", "blur", "occlusion"):
        ax[2].plot(h.epoch + 1, h[f"val_ssim_{t}"], label=t)
    ax[2].set_title("Validation SSIM per corruption"); ax[2].set_xlabel("epoch"); ax[2].legend(frameon=False, ncol=2)
    fig.tight_layout(); fig.savefig(FIG / "curves_task1.pdf"); plt.close(fig)


def fig_curves_task2():
    c = hist("task2_classifier_history")
    fig, ax = plt.subplots(1, 3, figsize=(7.1, 1.9))
    ax[0].plot(c.epoch + 1, c.train_loss, label="train CE"); ax[0].plot(c.epoch + 1, c.val_ce, label="val CE")
    ax[0].set_title("Classifier: cross-entropy"); ax[0].set_xlabel("epoch"); ax[0].legend(frameon=False)
    ax[1].plot(c.epoch + 1, c.train_acc, label="train accuracy"); ax[1].plot(c.epoch + 1, c.val_accuracy, label="val accuracy")
    ax[1].plot(c.epoch + 1, c.val_macro_f1, "--", label="val macro-F1")
    ax[1].set_title("Classifier: accuracy / macro-F1"); ax[1].set_xlabel("epoch"); ax[1].set_ylim(0.6, 1.01); ax[1].legend(frameon=False)
    for t, col in (("salt_pepper", "#ef4444"), ("blur", "#0ea5e9"), ("occlusion", "#10b981")):
        v1 = hist(f"task2_specialist_{t}_history"); v2 = hist(f"task2_specialist_{t}_v2_history")
        ax[2].plot(v1.epoch + 1, v1.val_psnr, ":", color=col, lw=0.9)
        ax[2].plot(v2.epoch + 1, v2.val_psnr, color=col, label=t)
    ax[2].set_title("Specialists: validation PSNR (dotted = first run, α≈0.29)"); ax[2].set_xlabel("epoch"); ax[2].set_ylabel("dB"); ax[2].legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "curves_task2.pdf"); plt.close(fig)


def fig_curves_task3():
    v2 = pd.read_csv(ROOT / "reports/task3_v2/training_log_task3_v2.csv", comment="#")
    v1 = hist("task3_history")
    fig, ax = plt.subplots(1, 3, figsize=(7.1, 1.9))
    j2 = v2[v2.stage == "joint"]
    ax[0].plot(v1.epoch + 1, v1.val_obj, ":", color="#94a3b8", label="first run (α=0.32)"); ax[0].plot(j2.epoch, j2.val_obj, label="final (α=0.61)")
    ax[0].set_title("Joint fine-tuning: val objective"); ax[0].set_xlabel("epoch"); ax[0].legend(frameon=False)
    ax[1].plot(v1.epoch + 1, v1.val_gate_acc, ":", color="#94a3b8"); ax[1].plot(j2.epoch, j2.gate_acc)
    ax[1].set_title("Gate argmax accuracy (validation)"); ax[1].set_xlabel("epoch"); ax[1].set_ylim(0.95, 1.0)
    w = v2[v2.stage == "warmup"]
    ax[2].plot(v2.index + 1, v2.val_ssim, marker="o", ms=2)
    ax[2].axvline(len(w) + 0.5, color="#64748b", ls="--", lw=0.7); ax[2].text(len(w) + 0.8, v2.val_ssim.min() + 0.002, "joint →", fontsize=6)
    ax[2].text(0.7, v2.val_ssim.min() + 0.002, "gate-only", fontsize=6)
    ax[2].set_title("Validation SSIM (warm-up, then joint)"); ax[2].set_xlabel("epoch")
    fig.tight_layout(); fig.savefig(FIG / "curves_task3.pdf"); plt.close(fig)


def fig_curves_task4():
    h = hist("task4_history")
    fig, ax = plt.subplots(1, 3, figsize=(7.1, 1.9))
    ax[0].plot(h.epoch + 1, h.d_real, label="D real"); ax[0].plot(h.epoch + 1, h.d_fake, label="D fake")
    ax[0].set_title("Discriminator losses"); ax[0].set_xlabel("epoch"); ax[0].legend(frameon=False)
    ax[1].plot(h.epoch + 1, h.g_adv, label="G adversarial"); ax[1].plot(h.epoch + 1, h.g_l1, label="G L1 (train, [-1,1] scale)")
    ax[1].set_title("Generator losses"); ax[1].set_xlabel("epoch"); ax[1].legend(frameon=False)
    ax[2].plot(h.epoch + 1, h.val_ssim, label="val SSIM", color="#10b981"); ax2 = ax[2].twinx(); ax2.plot(h.epoch + 1, h.val_l1, label="val L1", color="#ef4444")
    ax2.spines["right"].set_visible(True); ax2.set_ylabel("val L1", color="#ef4444"); ax[2].set_ylabel("val SSIM", color="#10b981")
    ax[2].set_title("Validation (fixed 159 pairs)"); ax[2].set_xlabel("epoch")
    fig.tight_layout(); fig.savefig(FIG / "curves_task4.pdf"); plt.close(fig)


# ------------------------------------------------------------------------------------------ Optuna
def fig_optuna():
    studies = [("Task 1 (universal AE)", "task1", "min"), ("Task 2 classifier", "task2_classifier", "max"),
               ("Task 2 specialists (final)", "task2_specialists_v2", "min"), ("Task 3 (final)", "task3_v2", "min"),
               ("Task 4 cGAN", "task4", "min")]
    fig, ax = plt.subplots(1, 5, figsize=(7.1, 1.7))
    for a, (title, name, sense) in zip(ax, studies):
        d = pd.read_csv(ROOT / "optuna_studies" / f"{name}_trials.csv")
        comp, prun = d[d.state == "COMPLETE"], d[d.state == "PRUNED"]
        a.scatter(comp.number, comp.value, s=12, color="#4f46e5", label=f"complete ({len(comp)})", zorder=3)
        best = comp.value.cummin() if sense == "min" else comp.value.cummax()
        a.step(comp.number, best, where="post", color="#10b981", lw=0.9, label="best so far")
        if len(prun):
            y0 = a.get_ylim()[0]
            a.scatter(prun.number, [y0] * len(prun), marker="x", s=10, color="#ef4444", label=f"pruned ({len(prun)})", zorder=3)
        a.set_title(title + (" [higher = better]" if sense == "max" else ""), fontsize=6.0)
        a.set_xlabel("trial"); a.legend(frameon=False, fontsize=5.3, loc="best")
    ax[0].set_ylabel("validation objective")
    fig.tight_layout(); fig.savefig(FIG / "optuna_overview.pdf"); plt.close(fig)


# ------------------------------------------------------------------------------------------ data / design figures
def fig_corruptions():
    import torch
    from src.data.corruptions import apply_spec
    entries = json.load(open(ROOT / "manifests/test_manifest_full.json"))["entries"]
    name = "Maine_Coon_2"
    img = Image.open(ROOT / f"data/oxford-iiit-pet/images/{name}.jpg").convert("RGB").resize((128, 128), Image.Resampling.BICUBIC)
    clean = torch.from_numpy(np.asarray(img, np.float32).transpose(2, 0, 1) / 255)
    sel = [e for e in entries if e["image"] == name]
    titles = {"clean/none": "clean", "salt_pepper/low": "S&P p=0.03", "salt_pepper/medium": "S&P p=0.08", "salt_pepper/high": "S&P p=0.15",
              "blur/low": "blur k3 σ0.7", "blur/medium": "blur k5 σ1.5", "blur/high": "blur k7 σ2.5",
              "occlusion/low": "1 rect ≈10%", "occlusion/medium": "2 rects ≈20%", "occlusion/high": "3 rects ≈35%"}
    fig, ax = plt.subplots(1, 10, figsize=(7.1, 0.95))
    for a, e in zip(ax, sel):
        a.imshow(apply_spec(clean, e).permute(1, 2, 0).numpy()); a.axis("off")
        a.set_title(titles[f"{e['type']}/{e['severity']}"], fontsize=5.2)
    fig.subplots_adjust(wspace=0.04); fig.savefig(FIG / "corruption_grid.pdf"); plt.close(fig)


def fig_task4_progress():
    d = ROOT / "reports/task4"
    fig, ax = plt.subplots(1, 3, figsize=(7.1, 3.1))
    for a, ep in zip(ax, (1, 20, 100)):
        a.imshow(Image.open(d / f"samples_ep{ep:03d}.png")); a.axis("off"); a.set_title(f"epoch {ep}", fontsize=7)
    fig.subplots_adjust(wspace=0.02); fig.savefig(FIG / "task4_progress.pdf", dpi=200); plt.close(fig)


def stitch_montage():
    base = ROOT / "stitch_design"
    v1 = base / "stitch_restoration_studio_ai"
    v2 = base / "v2" / "stitch_restoration_studio_ai"
    rows = [("First prompt", [v1 / "universal_restoration" / "screen.png", v1 / "hard_routed_restoration" / "screen.png",
                              v1 / "soft_mixture_of_experts" / "screen.png", v1 / "face_to_sketch_generator" / "screen.png"]),
            ("Second prompt (layout revision)", [v2 / "universal_restoration_2" / "screen.png", v2 / "hard_routed_restoration_2" / "screen.png",
                                                 v2 / "soft_mixture_of_experts_2" / "screen.png", v2 / "face_to_sketch_generator_2" / "screen.png"])]
    fig, ax = plt.subplots(2, 4, figsize=(7.1, 3.6))
    for r, (title, paths) in enumerate(rows):
        for c, p in enumerate(paths):
            ax[r, c].axis("off")
            try:
                ax[r, c].imshow(Image.open(p))
            except Exception:
                ax[r, c].text(0.5, 0.5, "image not exported\nby Stitch (round 2)", ha="center", va="center", fontsize=6, color="#64748b")
            ax[r, c].set_title(["Universal", "Hard-routed", "Soft MoE", "Sketch"][c] if r == 0 else "", fontsize=6.5)
        ax[r, 0].text(-0.02, 0.5, title, transform=ax[r, 0].transAxes, rotation=90, ha="right", va="center", fontsize=6.5)
    fig.subplots_adjust(wspace=0.02, hspace=0.04); fig.savefig(FIG / "stitch_montage.pdf", dpi=170); plt.close(fig)


if __name__ == "__main__":
    for fn in (fig_arch_ae, fig_arch_routing, fig_arch_cgan, fig_arch_system, fig_curves_task1, fig_curves_task2, fig_curves_task3,
               fig_curves_task4, fig_optuna, fig_corruptions, fig_task4_progress, stitch_montage):
        fn(); print("ok", fn.__name__)
