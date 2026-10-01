"""Stage 3 of the report build: qualitative figures produced by running the FINAL ONNX models.

  python report/build_qualitative.py       (needs reports/final_full_test/per_image_metrics.csv.gz to pick failure cases)

  examples_{universal,hard,soft}.pdf : 12 representative test inputs (every condition), clean | input | output | abs. error
  failure_{universal,hard,soft}.pdf  : the 4 lowest-SSIM corrupted inputs of each system
  misroute_hard.pdf                  : inputs the classifier mis-routed, with predicted-route vs oracle-route output
  soft_weights_examples.pdf          : inputs where one expert dominates vs inputs where the weights are spread out
"""
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.eval.onnx_full_eval import load_image, load_sessions, run_batch  # noqa: E402

FIG = ROOT / "report" / "figures"
plt.rcParams.update({"font.size": 6.5, "figure.dpi": 150, "savefig.bbox": "tight"})
TYPES = ["clean", "salt_pepper", "blur", "occlusion"]
SHORT = {"clean": "clean", "salt_pepper": "S&P", "blur": "blur", "occlusion": "occl."}
PRETTY = {"t1": "universal", "hard_pred": "hard-routed", "soft": "soft MoE"}
IMAGES = ROOT / "data/oxford-iiit-pet/images"
ENTRIES = {e["idx"]: e for e in json.load(open(ROOT / "manifests/test_manifest_full.json"))["entries"]}
SESS = None


def sess():
    global SESS
    SESS = SESS or load_sessions(ROOT / "onnx_models", threads=2)
    return SESS


def run(ids):
    ents = [ENTRIES[i] for i in ids]
    clean = np.stack([load_image(IMAGES, e["image"]) for e in ents])
    df, outs = run_batch(sess(), clean, ents)
    return clean, df, outs


def show(ax, chw, cmap=None, vmax=None):
    a = np.asarray(chw)
    a = a.transpose(1, 2, 0) if a.ndim == 3 else a
    ax.imshow(np.clip(a, 0, 1), cmap=cmap, vmin=0 if cmap else None, vmax=vmax)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def err(out, clean):
    return np.abs(np.clip(out, 0, 1) - clean).mean(0)


def badge(ax, text):
    ax.text(0.04, 0.05, text, transform=ax.transAxes, color="white", fontsize=5, va="bottom",
            bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.65, lw=0))


def cond(e):
    return "clean" if e["type"] == "clean" else f"{SHORT[e['type']]} {e['severity']}"


def examples(key):
    ids = [0 * 10 + 0, 1 * 10 + 1, 2 * 10 + 2, 3 * 10 + 3, 4 * 10 + 4, 5 * 10 + 5, 6 * 10 + 6, 7 * 10 + 7, 8 * 10 + 8, 9 * 10 + 9, 10 * 10 + 3, 11 * 10 + 9]
    clean, df, outs = run(ids)
    out = outs[key]
    fig = plt.figure(figsize=(7.1, 6.1))
    gs = fig.add_gridspec(6, 8, wspace=0.03, hspace=0.06)
    for k, i in enumerate(ids):
        r, b = k % 6, k // 6
        for c, (title, img, kw) in enumerate([("clean target", clean[k], {}), ("input", outs["input"][k], {}),
                                              (f"{PRETTY[key]} output", out[k], {}), ("abs. error", err(out[k], clean[k]), dict(cmap="magma", vmax=0.4))]):
            ax = fig.add_subplot(gs[r, b * 4 + c])
            show(ax, img, **kw)
            if r == 0:
                ax.set_title(title, fontsize=6)
            if c == 0:
                badge(ax, cond(ENTRIES[i]))
            if c == 2:
                badge(ax, f"SSIM {df[key + '_ssim'].iloc[k]:.2f}")
    fig.savefig(FIG / f"examples_{'universal' if key == 't1' else key.replace('_pred', '')}.pdf"); plt.close(fig)


def failures(df_full, key, fname):
    d = df_full[df_full.type != "clean"].nsmallest(4, f"{key}_ssim")
    ids = d.idx.tolist()
    clean, df, outs = run(ids)
    fig, ax = plt.subplots(4, 4, figsize=(3.4, 3.5))
    for k in range(4):
        for c, (title, img, kw) in enumerate([("clean", clean[k], {}), ("input", outs["input"][k], {}), ("output", outs[key][k], {}),
                                              ("abs. error", err(outs[key][k], clean[k]), dict(cmap="magma", vmax=0.4))]):
            show(ax[k, c], img, **kw)
            if k == 0:
                ax[k, c].set_title(title, fontsize=6)
        badge(ax[k, 0], cond(ENTRIES[ids[k]])); badge(ax[k, 2], f"SSIM {df[key + '_ssim'].iloc[k]:.2f}")
    fig.subplots_adjust(wspace=0.03, hspace=0.05); fig.savefig(FIG / fname); plt.close(fig)
    return d


def misroutes(df_full):
    bad = df_full[df_full.label != df_full.pred].copy()
    bad["loss"] = bad.hard_oracle_ssim - bad.hard_pred_ssim
    d = bad.nlargest(4, "loss")
    ids = d.idx.tolist()
    clean, df, outs = run(ids)
    fig, ax = plt.subplots(4, 4, figsize=(3.4, 3.5))
    for k in range(4):
        e = ENTRIES[ids[k]]
        for c, (title, img) in enumerate([("clean", clean[k]), ("input", outs["input"][k]), ("predicted route", outs["hard_pred"][k]), ("oracle route", outs["hard_oracle"][k])]):
            show(ax[k, c], img)
            if k == 0:
                ax[k, c].set_title(title, fontsize=6)
        badge(ax[k, 0], f"true {SHORT[e['type']]}"); badge(ax[k, 1], f"pred. {SHORT[TYPES[int(df.pred.iloc[k])]]}")
        badge(ax[k, 2], f"SSIM {df.hard_pred_ssim.iloc[k]:.2f}"); badge(ax[k, 3], f"SSIM {df.hard_oracle_ssim.iloc[k]:.2f}")
    fig.subplots_adjust(wspace=0.03, hspace=0.05); fig.savefig(FIG / "misroute_hard.pdf"); plt.close(fig)
    return d


def soft_weight_examples(df_full):
    W = [f"w_{t}" for t in TYPES]
    wv = df_full[W].values
    ent = -(np.clip(wv, 1e-12, 1) * np.log(np.clip(wv, 1e-12, 1))).sum(1)
    d = df_full.assign(ent=ent, wmax=wv.max(1))
    # most "one-expert" (lowest entropy) input of each corruption type, and the three most spread-out corrupted inputs
    dom = [int(d[d.type == t].nsmallest(1, "ent").idx.iloc[0]) for t in ("salt_pepper", "blur", "occlusion")]
    spread = d[d.type != "clean"].nlargest(3, "ent").idx.astype(int).tolist()
    ids = dom + spread
    clean, dfx, outs = run(ids)
    fig = plt.figure(figsize=(7.1, 3.2))
    gs = fig.add_gridspec(3, 6, wspace=0.42, hspace=0.25, width_ratios=[1, 1, 1.25, 1, 1, 1.25])
    cols = ["#64748b", "#ef4444", "#0ea5e9", "#10b981"]
    for k, i in enumerate(ids):
        r, b = k % 3, k // 3
        a0, a1, a2 = (fig.add_subplot(gs[r, b * 3 + c]) for c in range(3))
        show(a0, outs["input"][k]); show(a1, outs["soft"][k])
        w = dfx[W].iloc[k].values
        a2.barh(range(4), w, color=cols); a2.set_xlim(0, 1); a2.set_yticks(range(4)); a2.set_yticklabels(["identity", "S&P", "blur", "occl."], fontsize=5.5)
        a2.invert_yaxis(); a2.tick_params(axis="x", labelsize=5)
        for s in ("top", "right"):
            a2.spines[s].set_visible(False)
        badge(a0, cond(ENTRIES[i])); badge(a1, f"SSIM {dfx.soft_ssim.iloc[k]:.2f}")
        if r == 0:
            a0.set_title("input", fontsize=6); a1.set_title("soft MoE output", fontsize=6)
            a2.set_title("one expert dominates" if b == 0 else "weights spread out", fontsize=6.5)
    fig.savefig(FIG / "soft_weights_examples.pdf"); plt.close(fig)


def main():
    df_full = pd.read_csv(ROOT / os.environ.get("EVAL_DIR", "reports/final_full_test") / "per_image_metrics.csv.gz")
    for key in ("t1", "hard_pred", "soft"):
        examples(key); print("examples", key)
    failures(df_full, "t1", "failure_universal.pdf"); failures(df_full, "hard_pred", "failure_hard.pdf"); failures(df_full, "soft", "failure_soft.pdf")
    print("failures")
    misroutes(df_full); print("misroutes")
    soft_weight_examples(df_full); print("soft weight examples")


if __name__ == "__main__":
    main()
