"""Stage 2 of the report build: tables, number macros and result figures from the FULL-protocol evaluation.

  python -m src.eval.onnx_full_eval            # once (about 1.5 h on CPU) -> reports/final_full_test/per_image_metrics.csv.gz
  python report/build_tables.py                # writes report/generated/*.tex and report/figures/*.pdf

Nothing in the report's result tables is typed by hand: every cell comes from the per-image evaluation file
or from the saved Optuna / Kaggle result files.  PSNR is clipped at 60 dB (identical images have infinite PSNR).
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
GEN, FIG = ROOT / "report" / "generated", ROOT / "report" / "figures"
GEN.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "legend.fontsize": 6.5, "xtick.labelsize": 6.5,
                     "ytick.labelsize": 6.5, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight"})

TYPES = ["clean", "salt_pepper", "blur", "occlusion"]
NICE = {"clean": "clean", "salt_pepper": "salt-and-pepper", "blur": "Gaussian blur", "occlusion": "occlusion"}
CONDS = [("clean", "none")] + [(t, s) for t in TYPES[1:] for s in ("low", "medium", "high")]
SETTING = {("clean", "none"): "--", ("salt_pepper", "low"): "$p{=}0.03$", ("salt_pepper", "medium"): "$p{=}0.08$", ("salt_pepper", "high"): "$p{=}0.15$",
           ("blur", "low"): "$k{=}3,\\sigma{=}0.7$", ("blur", "medium"): "$k{=}5,\\sigma{=}1.5$", ("blur", "high"): "$k{=}7,\\sigma{=}2.5$",
           ("occlusion", "low"): "1 rect, 10\\%", ("occlusion", "medium"): "2 rects, 20\\%", ("occlusion", "high"): "3 rects, 35\\%"}
NUMS = {}


def num(name, value, fmt="{:.3f}"):
    NUMS[name] = fmt.format(value) if not isinstance(value, str) else value


def write(name, text):
    (GEN / name).write_text(text, encoding="utf-8")


def psnr_c(x):
    return np.minimum(x, 60.0)


def load_df():
    import os
    df = pd.read_csv(ROOT / os.environ.get("EVAL_DIR", "reports/final_full_test") / "per_image_metrics.csv.gz")
    for m in ("input", "t1", "hard_pred", "hard_oracle", "soft"):
        df[f"{m}_psnr"] = psnr_c(df[f"{m}_psnr"])
    return df


def f3(x):
    return f"{x:.3f}"


# ------------------------------------------------------------------------------------------------ Task 1
def table_task1(df):
    rows = []
    for (t, s) in CONDS:
        d = df[(df.type == t) & (df.severity == s)]
        rows.append(f"{NICE[t]} & {SETTING[(t, s)]} & {s if s != 'none' else '--'} & {f3(d.input_ssim.mean())} & {f3(d.t1_l1.mean())} & "
                    f"{f3(d.t1_ssim.mean())}$\\pm${f3(d.t1_ssim.std())} & {d.t1_psnr.mean():.2f} \\\\")
    corr = df[df.type != "clean"]
    rows.append("\\midrule")
    rows.append(f"\\multicolumn{{3}}{{l}}{{all corrupted ({len(corr):,})}} & {f3(corr.input_ssim.mean())} & {f3(corr.t1_l1.mean())} & {f3(corr.t1_ssim.mean())} & {corr.t1_psnr.mean():.2f} \\\\")
    rows.append(f"\\multicolumn{{3}}{{l}}{{all conditions ({len(df):,})}} & {f3(df.input_ssim.mean())} & {f3(df.t1_l1.mean())} & {f3(df.t1_ssim.mean())} & {df.t1_psnr.mean():.2f} \\\\")
    write("tab_task1.tex", "\\begin{tabular}{llccccr}\n\\toprule\nCorruption & Setting & Level & Input SSIM & L1 & SSIM & PSNR (dB) \\\\\n\\midrule\n"
          + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    for t in TYPES:
        d = df[df.type == t]
        num(f"TOneSSIM{t.replace('_', '')}", d.t1_ssim.mean()); num(f"InputSSIM{t.replace('_', '')}", d.input_ssim.mean())
    num("TOneSSIMall", df.t1_ssim.mean()); num("TOnePSNRcorr", corr.t1_psnr.mean(), "{:.2f}"); num("InputSSIMall", df.input_ssim.mean())
    num("TOneSSIMcorr", corr.t1_ssim.mean()); num("InputSSIMcorr", corr.input_ssim.mean())


# ------------------------------------------------------------------------------------------------ Task 2
def classifier_stats(df):
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
    y, p = df.label.values, df.pred.values
    acc = float((y == p).mean())
    P, R, F, S = precision_recall_fscore_support(y, p, labels=[0, 1, 2, 3], zero_division=0)
    mp, mr, mf, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
    cm = confusion_matrix(y, p, labels=[0, 1, 2, 3])
    return acc, (P, R, F, S), (mp, mr, mf), cm


def table_classifier(df):
    acc, (P, R, F, S), (mp, mr, mf), cm = classifier_stats(df)
    rows = [f"{NICE[t]} & {S[i]:,} & {P[i]:.3f} & {R[i]:.3f} & {F[i]:.3f} \\\\" for i, t in enumerate(TYPES)]
    rows += ["\\midrule", f"macro average & {S.sum():,} & {mp:.3f} & {mr:.3f} & {mf:.3f} \\\\",
             f"\\multicolumn{{2}}{{l}}{{overall accuracy}} & \\multicolumn{{3}}{{c}}{{{acc:.4f}}} \\\\"]
    write("tab_classifier.tex", "\\begin{tabular}{lrccc}\n\\toprule\nClass & Support & Precision & Recall & F1 \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    num("ClfAcc", acc, "{:.4f}"); num("ClfMacroF", mf, "{:.4f}"); num("ClfMacroP", mp, "{:.4f}"); num("ClfMacroR", mr, "{:.4f}")
    num("Misroute", 100 * (1 - acc), "{:.2f}"); num("MisrouteCount", f"{int((df.label != df.pred).sum()):,}")
    fig, ax = plt.subplots(figsize=(3.3, 2.9))
    cmn = cm / cm.sum(1, keepdims=True)
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(4)); ax.set_yticks(range(4)); ax.set_xticklabels([NICE[t].replace("salt-and-pepper", "salt&pepper") for t in TYPES], rotation=30, ha="right")
    ax.set_yticklabels([NICE[t].replace("salt-and-pepper", "salt&pepper") for t in TYPES]); ax.set_xlabel("predicted"); ax.set_ylabel("true")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cmn[i, j]:.3f}", ha="center", va="center", fontsize=7, color="white" if cmn[i, j] > 0.6 else "black")
    ax.set_title("Normalised confusion matrix (full test protocol)"); fig.colorbar(im, fraction=0.046)
    fig.savefig(FIG / "confusion_matrix.pdf"); plt.close(fig)
    return cm


def table_routing(df):
    rows = []
    for t in TYPES:
        d = df[df.type == t]
        mis = 100 * (d.label != d.pred).mean()
        rows.append(f"{NICE[t]} & {f3(d.input_ssim.mean())} & {f3(d.t1_ssim.mean())} & {f3(d.hard_oracle_ssim.mean())} & {f3(d.hard_pred_ssim.mean())} & "
                    f"{d.hard_oracle_l1.mean():.4f} & {d.hard_pred_l1.mean():.4f} & {mis:.2f} \\\\")
    rows.append("\\midrule")
    rows.append(f"all ({len(df):,}) & {f3(df.input_ssim.mean())} & {f3(df.t1_ssim.mean())} & {f3(df.hard_oracle_ssim.mean())} & {f3(df.hard_pred_ssim.mean())} & "
                f"{df.hard_oracle_l1.mean():.4f} & {df.hard_pred_l1.mean():.4f} & {100 * (df.label != df.pred).mean():.2f} \\\\")
    write("tab_routing.tex", "\\begin{tabular}{lccccccc}\n\\toprule\n & \\multicolumn{3}{c}{SSIM} & & \\multicolumn{2}{c}{L1} & \\\\\n\\cmidrule(lr){2-5}\\cmidrule(lr){6-7}\n"
          "Input type & no restoration & universal & oracle & predicted & oracle & predicted & misroute \\% \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    num("HardOracleSSIM", df.hard_oracle_ssim.mean()); num("HardPredSSIM", df.hard_pred_ssim.mean())
    num("HardPredL1", df.hard_pred_l1.mean(), "{:.4f}"); num("HardOracleL1", df.hard_oracle_l1.mean(), "{:.4f}")
    for t in TYPES:
        d = df[df.type == t]
        num(f"HardSSIM{t.replace('_', '')}", d.hard_pred_ssim.mean())
    # cost of classifier errors: SSIM lost by mis-routed images relative to oracle routing
    bad = df[df.label != df.pred]
    num("MisrouteSSIMLoss", (bad.hard_oracle_ssim - bad.hard_pred_ssim).mean(), "{:.3f}")
    num("MisrouteSystemLoss", (df.hard_oracle_ssim - df.hard_pred_ssim).mean(), "{:.4f}")


# ------------------------------------------------------------------------------------------------ Task 3
def table_task3(df):
    rows = []
    for (t, s) in CONDS:
        d = df[(df.type == t) & (df.severity == s)]
        rows.append(f"{NICE[t]} & {s if s != 'none' else '--'} & {f3(d.t1_ssim.mean())} & {f3(d.hard_pred_ssim.mean())} & {f3(d.soft_ssim.mean())}$\\pm${f3(d.soft_ssim.std())} & "
                    f"{d.soft_l1.mean():.4f} & {d.soft_psnr.mean():.2f} \\\\")
    rows.append("\\midrule")
    rows.append(f"all ({len(df):,}) & & {f3(df.t1_ssim.mean())} & {f3(df.hard_pred_ssim.mean())} & {f3(df.soft_ssim.mean())}$\\pm${f3(df.soft_ssim.std())} & {df.soft_l1.mean():.4f} & {df.soft_psnr.mean():.2f} \\\\")
    write("tab_task3.tex", "\\begin{tabular}{llcccrr}\n\\toprule\n & & \\multicolumn{3}{c}{SSIM} & \\multicolumn{2}{c}{soft MoE} \\\\\n\\cmidrule(lr){3-5}\\cmidrule(lr){6-7}\n"
          "Corruption & Level & universal & hard (pred.) & soft MoE & L1 & PSNR (dB) \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    num("SoftSSIM", df.soft_ssim.mean()); num("SoftL1", df.soft_l1.mean(), "{:.4f}")
    for t in TYPES:
        num(f"SoftSSIM{t.replace('_', '')}", df[df.type == t].soft_ssim.mean())
    # gate behaviour
    W = [f"w_{t}" for t in TYPES]
    rows, heat = [], []
    for (t, s) in CONDS:
        d = df[(df.type == t) & (df.severity == s)]
        w = d[W].mean().values; heat.append(w)
        rows.append(f"{NICE[t]} {s if s != 'none' else ''} & " + " & ".join(f"{v:.3f}" for v in w) + f" & {NICE[TYPES[int(np.argmax(w))]]} \\\\")
    write("tab_gate.tex", "\\begin{tabular}{lcccc l}\n\\toprule\nTrue condition & $w_0$ identity & $w_1$ salt & $w_2$ blur & $w_3$ occl. & dominant \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    heat = np.array(heat)
    fig, ax = plt.subplots(figsize=(3.4, 3.1))
    im = ax.imshow(heat, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(4)); ax.set_xticklabels(["identity", "salt&pepper", "blur", "occlusion"], rotation=25, ha="right")
    ax.set_yticks(range(len(CONDS))); ax.set_yticklabels([f"{t} {s}".replace("salt_pepper", "S&P").replace(" none", "") for t, s in CONDS])
    for i in range(heat.shape[0]):
        for j in range(4):
            ax.text(j, i, f"{heat[i, j]:.2f}", ha="center", va="center", fontsize=6, color="white" if heat[i, j] < 0.55 else "black")
    ax.set_title("Mean gate weight per true condition"); fig.colorbar(im, fraction=0.046)
    fig.savefig(FIG / "gate_heatmap.pdf"); plt.close(fig)
    # dominance / dead-expert analysis
    wv = df[W].values
    top = wv.argmax(1)
    rows = []
    for t in TYPES:
        m = (df.type == t).values
        share = [(top[m] == k).mean() for k in range(4)]
        rows.append(f"{NICE[t]} & " + " & ".join(f"{100 * v:.1f}" for v in share) + f" & {np.mean(wv[m].max(1)):.3f} \\\\")
    rows.append("\\midrule")
    rows.append("all inputs & " + " & ".join(f"{100 * (top == k).mean():.1f}" for k in range(4)) + f" & {np.mean(wv.max(1)):.3f} \\\\")
    write("tab_dominance.tex", "\\begin{tabular}{lccccc}\n\\toprule\nTrue type & \\multicolumn{4}{c}{\\% of inputs where branch has the largest weight} & mean max weight \\\\\n"
          "\\cmidrule(lr){2-5}\n & identity & salt & blur & occl. & \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    ent = -(np.clip(wv, 1e-12, 1) * np.log(np.clip(wv, 1e-12, 1))).sum(1)
    num("GateEntropyMean", ent.mean(), "{:.2f}"); num("GateMaxMean", wv.max(1).mean())
    for k, t in enumerate(TYPES):
        num(f"TopShare{t.replace('_', '')}", 100 * (top == k).mean(), "{:.1f}")
    num("HomeWeightSalt", df[df.type == "salt_pepper"].w_salt_pepper.mean(), "{:.2f}")
    num("HomeWeightBlur", df[df.type == "blur"].w_blur.mean(), "{:.2f}")
    num("HomeWeightOcc", df[df.type == "occlusion"].w_occlusion.mean(), "{:.2f}")
    num("HomeWeightClean", df[df.type == "clean"].w_clean.mean(), "{:.2f}")
    return ent


def bar_comparison(df):
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    x = np.arange(4); wd = 0.2
    for k, (col, lab, c) in enumerate([("input_ssim", "no restoration", "#94a3b8"), ("t1_ssim", "universal (Task 1)", "#6366f1"),
                                       ("hard_pred_ssim", "hard-routed (Task 2)", "#10b981"), ("soft_ssim", "soft MoE (Task 3)", "#f59e0b")]):
        ax.bar(x + (k - 1.5) * wd, [df[df.type == t][col].mean() for t in TYPES], wd, label=lab, color=c)
    ax.set_xticks(x); ax.set_xticklabels([NICE[t].replace("salt-and-pepper", "salt&pepper") for t in TYPES]); ax.set_ylabel("mean test SSIM"); ax.set_ylim(0, 1.08)
    ax.legend(ncol=2, frameon=False, fontsize=5.8, loc="upper center"); ax.set_title("SSIM by input type (36,690 test inputs)")
    fig.savefig(FIG / "bar_comparison.pdf"); plt.close(fig)


# ------------------------------------------------------------------------------------------------ Optuna / static tables
def optuna_tables():
    def best(name, sense="min"):
        d = pd.read_csv(ROOT / "optuna_studies" / f"{name}_trials.csv")
        c = d[d.state == "COMPLETE"]
        b = c.loc[c.value.idxmin() if sense == "min" else c.value.idxmax()]
        return len(d), len(c), int((d.state == "PRUNED").sum()), b
    rows = []
    for label, name, sense in [("Task 1 universal AE", "task1", "min"), ("Task 2 classifier", "task2_classifier", "max"),
                               ("Task 2 specialists (first run)", "task2_specialists", "min"), ("Task 2 specialists (final, $\\alpha\\geq0.55$)", "task2_specialists_v2", "min"),
                               ("Task 3 soft MoE (first run)", "task3", "min"), ("Task 3 soft MoE (final, $\\alpha\\geq0.55$)", "task3_v2", "min"),
                               ("Task 4 cGAN (selected)", "task4", "min"), ("Task 4 cGAN (rejected variant)", "task4_v2", "min")]:
        n, c, p, b = best(name, sense)
        rows.append(f"{label} & {n} & {c} & {p} & \\#{int(b['number'])} & {b['value']:.4f} \\\\")
    write("tab_optuna_counts.tex", "\\begin{tabular}{lrrrrr}\n\\toprule\nStudy & Trials & Completed & Pruned & Best trial & Best value \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    sel = {n: json.load(open(ROOT / "configs" / f"{n}.json")) for n in ("task1_best", "task2_classifier_best", "task2_specialists_best_v2", "task3_best_v2", "task4_best")}
    fmt = lambda v: (f"{v:.2e}" if isinstance(v, float) and (v < 0.01) else (f"{v:.3f}" if isinstance(v, float) else str(v)))
    lines = []
    for k, label in [("task1_best", "Task 1"), ("task2_classifier_best", "Classifier"), ("task2_specialists_best_v2", "Specialists"), ("task3_best_v2", "Soft MoE"), ("task4_best", "cGAN")]:
        lines.append(f"{label} & " + ", ".join(f"{a.replace('_', '\\_')}={fmt(b)}" for a, b in sel[k].items()) + " \\\\")
    write("tab_selected.tex", "\\begin{tabular}{lp{6.3cm}}\n\\toprule\nModel & Selected configuration \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


def table_v1_v2():
    """First run (alpha ~ 0.29/0.32) vs final run (alpha >= 0.55) on the single-condition Kaggle test manifest (3,669 inputs)."""
    r1 = json.load(open(ROOT / "reports/task2_routing/routing_summary.json")); r2 = json.load(open(ROOT / "reports/task2_routing_v2/routing_summary.json"))
    t1 = pd.read_csv(ROOT / "reports/task3/per_image_metrics.csv"); t2 = pd.read_csv(ROOT / "reports/task3_v2/per_image_metrics.csv")
    rows = []
    for lab, a, b in [("Hard routing (oracle)", r1["oracle"]["overall"], r2["oracle"]["overall"]), ("Hard routing (predicted)", r1["predicted"]["overall"], r2["predicted"]["overall"])]:
        rows.append(f"{lab} & {a['l1']:.4f} & {b['l1']:.4f} & {a['ssim']:.4f} & {b['ssim']:.4f} \\\\")
    rows.append(f"Soft MoE & {t1.l1.mean():.4f} & {t2.l1.mean():.4f} & {t1.ssim.mean():.4f} & {t2.ssim.mean():.4f} \\\\")
    write("tab_v1_v2.tex", "\\begin{tabular}{lcccc}\n\\toprule\n & \\multicolumn{2}{c}{L1 (lower is better)} & \\multicolumn{2}{c}{SSIM} \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
          "System & first run & final & first run & final \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    num("SoftL1First", t1.l1.mean(), "{:.4f}"); num("SoftL1Final", t2.l1.mean(), "{:.4f}")
    num("HardPredL1First", r1["predicted"]["overall"]["l1"], "{:.4f}"); num("HardPredL1Final", r2["predicted"]["overall"]["l1"], "{:.4f}")
    num("L1DropPct", 100 * (1 - r2["predicted"]["overall"]["l1"] / r1["predicted"]["overall"]["l1"]), "{:.0f}")


def table_task4():
    m = json.load(open(ROOT / "reports/task4_eval/metrics_per_style.json")); m2 = json.load(open(ROOT / "reports/task4_eval_v2/metrics_per_style.json"))
    rows = []
    for s in ("1", "2", "3"):
        d = m["per_style"][s]
        rows.append(f"Style {s} & {d['l1_count']} & {d['l1_mean']:.4f} & {d['ssim_mean']:.3f}$\\pm${d['ssim_std']:.3f} & {d['psnr_mean']:.2f} \\\\")
    rows += ["\\midrule", f"all (selected model) & {m['n']} & {m['l1']:.4f} & {m['ssim']:.3f} & {m['psnr']:.2f} \\\\",
             f"all (rejected variant) & {m2['n']} & {m2['l1']:.4f} & {m2['ssim']:.3f} & {m2['psnr']:.2f} \\\\"]
    write("tab_task4.tex", "\\begin{tabular}{lrccc}\n\\toprule\nTest subset & Pairs & L1 & SSIM & PSNR (dB) \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    num("FourSSIM", m["ssim"]); num("FourL1", m["l1"], "{:.4f}"); num("FourPSNR", m["psnr"], "{:.2f}")
    num("FourSSIMv", m2["ssim"]); num("FourL1v", m2["l1"], "{:.4f}")
    for s in ("1", "2", "3"):
        num(f"FourL1s{s}", m["per_style"][s]["l1_mean"], "{:.3f}"); num(f"FourSSIMs{s}", m["per_style"][s]["ssim_mean"], "{:.3f}")


def onnx_table():
    """File size and export-time PyTorch-vs-ONNX difference (from the export logs) + CPU latency (report/build_analysis.py)."""
    diffs = {"task1_universal_dae.onnx": "5.4e-7", "task2_classifier.onnx": "2.0e-6", "task2_specialist_salt_pepper.onnx": "5.7e-7",
             "task2_specialist_blur.onnx": "4.5e-7", "task2_specialist_occlusion.onnx": "5.4e-7", "task3_soft_moe.onnx": "2.4e-7",
             "task4_generator.onnx": "1.4e-6"}
    desc = {"task1_universal_dae.onnx": "Task 1 universal autoencoder", "task2_classifier.onnx": "Task 2 classifier",
            "task2_specialist_salt_pepper.onnx": "Task 2 specialist (salt-and-pepper)", "task2_specialist_blur.onnx": "Task 2 specialist (blur)",
            "task2_specialist_occlusion.onnx": "Task 2 specialist (occlusion)", "task3_soft_moe.onnx": "Task 3 soft mixture (whole pipeline)",
            "task4_generator.onnx": "Task 4 generator"}
    lat_file = GEN / "latency.json"
    lat = json.load(open(lat_file)) if lat_file.exists() else {}
    rows = []
    for f, d in desc.items():
        mb = (ROOT / "onnx_models" / f).stat().st_size / 1e6
        rows.append(f"{d} & {mb:.1f} & {diffs[f]} & {lat.get(f, '--')} \\\\")
    write("tab_onnx.tex", "\\begin{tabular}{lrrr}\n\\toprule\nModel & Size (MB) & Max $|$PyTorch$-$ONNX$|$ & Latency (ms) \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def static_numbers():
    g = json.load(open(ROOT / "reports/task1_generalization/generalization.json"))
    single = json.load(open(ROOT / "reports/task1/metrics_overall.json"))["overall"]
    num("TOneSSIMsingle", single["ssim"]); num("TOnePSNRsingle", single["psnr"], "{:.2f}")
    num("GapSSIM", g["gap_ssim"], "{:.3f}"); num("TrainSSIMsub", g["train_subset"]["ssim"]); num("ValSSIMsub", g["val"]["ssim"])
    a1 = json.load(open(ROOT / "reports/task1_attempt1/metrics_overall.json"))["overall"]
    num("AttemptOneSSIM", a1["ssim"]); num("AttemptOnePSNR", a1["psnr"], "{:.2f}")
    h = pd.read_json(ROOT / "checkpoints/task1_history.json")
    num("TOneBestEpoch", int(h.val_obj.idxmin()) + 1, "{:d}")


def table_task1_psnr(df):
    """Task 1 table with the input PSNR next to the input SSIM (SSIM and PSNR tell different stories for occlusion)."""
    def psnr_txt(v):
        return "$\\geq$60" if v >= 59.9 else f"{v:.2f}"
    rows = []
    for (t, s) in CONDS:
        d = df[(df.type == t) & (df.severity == s)]
        rows.append(f"{NICE[t]} & {SETTING[(t, s)]} & {s if s != 'none' else '--'} & {f3(d.input_ssim.mean())} & {psnr_txt(d.input_psnr.mean())} & "
                    f"{f3(d.t1_l1.mean())} & {f3(d.t1_ssim.mean())}$\\pm${f3(d.t1_ssim.std())} & {d.t1_psnr.mean():.2f} \\\\")
    corr = df[df.type != "clean"]
    rows.append("\\midrule")
    rows.append(f"\\multicolumn{{3}}{{l}}{{all corrupted ({len(corr):,})}} & {f3(corr.input_ssim.mean())} & {corr.input_psnr.mean():.2f} & {f3(corr.t1_l1.mean())} & "
                f"{f3(corr.t1_ssim.mean())} & {corr.t1_psnr.mean():.2f} \\\\")
    rows.append(f"\\multicolumn{{3}}{{l}}{{all conditions ({len(df):,})}} & {f3(df.input_ssim.mean())} & {df.input_psnr.mean():.2f} & {f3(df.t1_l1.mean())} & "
                f"{f3(df.t1_ssim.mean())} & {df.t1_psnr.mean():.2f} \\\\")
    write("tab_task1.tex", "\\begin{tabular}{llcccccr}\n\\toprule\n & & & \\multicolumn{2}{c}{Input (no restoration)} & \\multicolumn{3}{c}{Universal autoencoder} \\\\\n"
          "\\cmidrule(lr){4-5}\\cmidrule(lr){6-8}\nCorruption & Setting & Level & SSIM & PSNR & L1 & SSIM & PSNR \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def extra_numbers(df):
    """PSNR by type, the misroute breakdown, the SSIM-versus-PSNR behaviour on occlusion, and gate entropy by condition."""
    for t in TYPES[1:]:
        d = df[df.type == t]
        k = t.replace("_", "")
        num(f"InPSNR{k}", d.input_psnr.mean(), "{:.1f}"); num(f"TOnePSNR{k}", d.t1_psnr.mean(), "{:.1f}")
        num(f"HardPSNR{k}", d.hard_pred_psnr.mean(), "{:.1f}"); num(f"SoftPSNR{k}", d.soft_psnr.mean(), "{:.1f}")
    g = lambda t, s, col: df[(df.type == t) & (df.severity == s)][col].mean()
    num("GateOccLow", g("occlusion", "low", "w_occlusion"), "{:.2f}"); num("GateOccHigh", g("occlusion", "high", "w_occlusion"), "{:.2f}")
    # where is doing nothing better than restoring (by SSIM)?
    for sev in ("low", "medium", "high"):
        d = df[(df.type == "occlusion") & (df.severity == sev)]
        num(f"OccNoneBetter{sev.capitalize()}", 100 * (d.input_ssim > d.t1_ssim).mean(), "{:.0f}")
    # misroute breakdown table
    bad = df[df.label != df.pred].copy()
    bad["pred_name"] = bad.pred.map(dict(enumerate(TYPES)))
    bad["d_ssim"] = bad.hard_pred_ssim - bad.hard_oracle_ssim
    rows = []
    for (t, s, p), d in bad.groupby(["type", "severity", "pred_name"]):
        if len(d) >= 3:
            rows.append((len(d), f"{NICE[t]}{'' if s == 'none' else ' (' + s + ')'} & {NICE[p]} & {len(d)} & {d.d_ssim.mean():+.3f} \\\\"))
    rows.sort(key=lambda r: -r[0])
    n_small = int(sum(1 for (t, s, p), d in bad.groupby(['type', 'severity', 'pred_name']) if len(d) < 3) and sum(len(d) for _, d in bad.groupby(['type', 'severity', 'pred_name']) if len(d) < 3))
    lines = [r[1] for r in rows] + ([f"other (fewer than 3 each) & -- & {n_small} & -- \\\\"] if n_small else [])
    lines += ["\\midrule", f"all mis-routed inputs & -- & {len(bad)} & {bad.d_ssim.mean():+.3f} \\\\"]
    write("tab_misroute.tex", "\\begin{tabular}{llrr}\n\\toprule\nTrue input & Routed to & Count & Mean $\\Delta$SSIM vs.\\ oracle route \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    cl = bad[bad.type == "clean"]; oc = bad[bad.type == "occlusion"]
    num("MisClean", f"{len(cl)}"); num("MisCleanLoss", cl.d_ssim.mean(), "{:.2f}")
    num("MisOcc", f"{len(oc)}"); num("MisOccGain", oc.d_ssim.mean(), "{:+.2f}")
    # entropy of the gate weights by condition
    wv = df[[f"w_{t}" for t in TYPES]].values
    df = df.assign(ent=-(np.clip(wv, 1e-12, 1) * np.log(np.clip(wv, 1e-12, 1))).sum(1))
    num("EntOccHigh", df[(df.type == "occlusion") & (df.severity == "high")].ent.mean(), "{:.2f}")
    num("EntClean", df[df.type == "clean"].ent.mean(), "{:.2f}"); num("EntOccLow", df[(df.type == "occlusion") & (df.severity == "low")].ent.mean(), "{:.2f}")


def main():
    df = load_df()
    num("NEntries", f"{len(df):,}"); num("NImages", f"{df.image.nunique():,}")
    table_task1(df); table_task1_psnr(df); table_classifier(df); table_routing(df); table_task3(df); bar_comparison(df); extra_numbers(df)
    optuna_tables(); table_v1_v2(); table_task4(); static_numbers(); onnx_table()
    words = "Zero One Two Three Four Five Six Seven Eight Nine".split()
    macro = lambda k: "".join(words[int(ch)] if ch.isdigit() else ch for ch in k)  # LaTeX names cannot contain digits
    write("numbers.tex", "\n".join(f"\\newcommand{{\\n{macro(k)}}}{{{v}}}" for k, v in sorted(NUMS.items())) + "\n")
    print("wrote", len(NUMS), "number macros and the tables to", GEN)


if __name__ == "__main__":
    main()
