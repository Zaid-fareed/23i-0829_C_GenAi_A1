"""Standalone classifier evaluation on the test manifest.

  python -m src.eval.task2_classifier_eval --ckpt checkpoints/task2_classifier.pt --out_dir reports/task2_classifier

Outputs: metrics.json (accuracy, macro P/R/F1, per-class P/R/F1, raw+normalized confusion matrix),
        confusion_matrix.png (normalized, annotated).
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data.corruptions import TYPES
from src.data.pets import build_eval_loader
from src.models.routing import load_classifier
from src.train.engine_cls import classifier_metrics, predict_classifier
from src.utils import paths
from src.utils.common import get_device, seed_everything


def plot_confusion(cm_norm, out_path):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(4)); ax.set_xticklabels(TYPES, rotation=30, ha="right")
    ax.set_yticks(range(4)); ax.set_yticklabels(TYPES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("Normalized confusion matrix")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cm_norm[i][j]:.2f}", ha="center", va="center",
                    color="white" if cm_norm[i][j] > 0.5 else "black")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout(); plt.savefig(out_path, dpi=130); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(paths.checkpoint_dir() / "task2_classifier.pt"))
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--out_dir", default="reports/task2_classifier")
    a = ap.parse_args()
    seed_everything(42)
    device = get_device()
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    model, ck = load_classifier(a.ckpt, device)
    loader = build_eval_loader(a.split, batch_size=64)
    probs, labels, _ = predict_classifier(model, loader, device)
    metrics = classifier_metrics(probs, labels)

    json.dump({"split": a.split, "ckpt_params": ck.get("params"), **metrics}, open(out_dir / "metrics.json", "w"),
              indent=2)
    plot_confusion(np.array(metrics["confusion_matrix_normalized"]), out_dir / "confusion_matrix.png")

    print(f"accuracy={metrics['accuracy']:.4f}  macro_P={metrics['macro_precision']:.4f} "
          f"macro_R={metrics['macro_recall']:.4f}  macro_F1={metrics['macro_f1']:.4f}")
    for t in TYPES:
        pc = metrics["per_class"][t]
        print(f"  {t:12s} P={pc['precision']:.3f} R={pc['recall']:.3f} F1={pc['f1']:.3f} n={pc['support']}")
    print("saved", out_dir / "metrics.json", out_dir / "confusion_matrix.png")


if __name__ == "__main__":
    main()
