"""Stage 4 of the report build: two small analyses whose numbers the report quotes.

  (a) colour preservation of the restoration models (first run vs final run), on salt-and-pepper test inputs;
  (b) sharpness (edge strength) of generated vs real sketches on the FS2K test set, and how it reacts to input sharpness.

  PYTHONPATH=. FS2K_DIR=data/fs2k_raw/FS2K python report/build_analysis.py
Writes report/generated/tab_colour.tex, tab_sharpness.tex and appends macros to report/generated/numbers_analysis.tex
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.corruptions import apply_spec  # noqa: E402
from src.eval.onnx_full_eval import MODELS, load_image  # noqa: E402

GEN = ROOT / "report" / "generated"
GEN.mkdir(parents=True, exist_ok=True)
MACROS = {}
WORDS = "Zero One Two Three Four Five Six Seven Eight Nine".split()


def macro(name, value):
    MACROS["".join(WORDS[int(c)] if c.isdigit() else c for c in name)] = value


def sessions(dir_map):
    opts = ort.SessionOptions(); opts.intra_op_num_threads = 4
    return {k: ort.InferenceSession(str(Path(d) / MODELS[k]), opts, providers=["CPUExecutionProvider"]) for k, d in dir_map.items()}


def colourfulness(x):
    return float((x.max(0) - x.min(0)).mean())


def colour_table(n=200):
    entries = [e for e in json.load(open(ROOT / "manifests/test_manifest_full.json"))["entries"] if e["type"] == "salt_pepper"][:n]
    final = sessions({"t1": ROOT / "onnx_models", "sp": ROOT / "onnx_models", "moe": ROOT / "onnx_models"})
    first = sessions({"sp": ROOT / "onnx_models_v1", "moe": ROOT / "onnx_models_v1"})
    res = {k: [] for k in ("clean", "input", "universal", "sp_first", "sp_final", "moe_first", "moe_final")}
    for e in entries:
        clean = load_image(ROOT / "data/oxford-iiit-pet/images", e["image"])
        x = apply_spec(torch.from_numpy(clean), e).numpy().astype(np.float32)[None]
        res["clean"].append(colourfulness(clean)); res["input"].append(colourfulness(x[0]))
        for key, s, name in (("universal", final["t1"], "image"), ("sp_first", first["sp"], "image"), ("sp_final", final["sp"], "image")):
            res[key].append(colourfulness(np.clip(s.run(None, {name: x})[0][0], 0, 1)))
        res["moe_first"].append(colourfulness(np.clip(first["moe"].run(None, {"image": x})[0][0], 0, 1)))
        res["moe_final"].append(colourfulness(np.clip(final["moe"].run(None, {"image": x})[0][0], 0, 1)))
    base = np.mean(res["clean"])
    rows = [("clean image (reference)", "clean"), ("corrupted input", "input"), ("universal AE (Task 1)", "universal"),
            ("S\\&P specialist, first run ($\\alpha\\approx0.29$)", "sp_first"), ("S\\&P specialist, final ($\\alpha\\approx0.60$)", "sp_final"),
            ("soft MoE, first run ($\\alpha\\approx0.32$)", "moe_first"), ("soft MoE, final ($\\alpha\\approx0.61$)", "moe_final")]
    lines = [f"{lab} & {np.mean(res[k]):.4f} & {100 * np.mean(res[k]) / base:.0f}\\% \\\\" for lab, k in rows]
    (GEN / "tab_colour.tex").write_text("\\begin{tabular}{lcc}\n\\toprule\nModel output & Colourfulness & Share of clean colour \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")
    for k in ("universal", "sp_first", "sp_final", "moe_first", "moe_final"):
        macro(f"Colour_{k}".replace("_", ""), f"{100 * np.mean(res[k]) / base:.0f}")
    macro("ColourN", str(len(entries)))
    print("colour table done", {k: round(100 * np.mean(v) / base) for k, v in res.items()})


def colour_table2(n=200):
    """Colour kept by each model on salt-and-pepper AND occlusion test inputs (first run vs final run of the specialists / mixture)."""
    all_entries = json.load(open(ROOT / "manifests/test_manifest_full.json"))["entries"]
    final = sessions({"t1": ROOT / "onnx_models", "sp": ROOT / "onnx_models", "bl": ROOT / "onnx_models", "oc": ROOT / "onnx_models", "moe": ROOT / "onnx_models"})
    first = sessions({"sp": ROOT / "onnx_models_v1", "bl": ROOT / "onnx_models_v1", "oc": ROOT / "onnx_models_v1", "moe": ROOT / "onnx_models_v1"})
    res = {}
    for ctype, key in (("salt_pepper", "sp"), ("occlusion", "oc")):
        entries = [e for e in all_entries if e["type"] == ctype][:n]
        r = {k: [] for k in ("clean", "input", "universal", "spec_first", "spec_final", "moe_first", "moe_final")}
        for e in entries:
            clean = load_image(ROOT / "data/oxford-iiit-pet/images", e["image"])
            x = apply_spec(torch.from_numpy(clean), e).numpy().astype(np.float32)[None]
            run = lambda s: colourfulness(np.clip(s.run(None, {"image": x})[0][0], 0, 1))
            r["clean"].append(colourfulness(clean)); r["input"].append(colourfulness(x[0]))
            r["universal"].append(run(final["t1"])); r["spec_first"].append(run(first[key])); r["spec_final"].append(run(final[key]))
            r["moe_first"].append(run(first["moe"])); r["moe_final"].append(run(final["moe"]))
        base = np.mean(r["clean"])
        res[ctype] = {k: 100 * np.mean(v) / base for k, v in r.items()}
    rows = [("clean image (reference)", "clean"), ("corrupted input", "input"), ("universal AE (Task 1)", "universal"),
            ("specialist, first run ($\\alpha\\approx0.3$)", "spec_first"), ("specialist, final ($\\alpha\\approx0.6$)", "spec_final"),
            ("soft MoE, first run ($\\alpha\\approx0.3$)", "moe_first"), ("soft MoE, final ($\\alpha\\approx0.6$)", "moe_final")]
    lines = [f"{lab} & {res['salt_pepper'][k]:.0f}\\% & {res['occlusion'][k]:.0f}\\% \\\\" for lab, k in rows]
    (GEN / "tab_colour.tex").write_text("\\begin{tabular}{lcc}\n\\toprule\nModel output & salt-and-pepper inputs & occlusion inputs \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")
    names = {"universal": "Universal", "spec_first": "SpecFirst", "spec_final": "SpecFinal", "moe_first": "MoeFirst", "moe_final": "MoeFinal"}
    for ctype, tag in (("salt_pepper", "Sp"), ("occlusion", "Oc")):
        for k, nm in names.items():
            macro(f"Col{tag}{nm}", f"{res[ctype][k]:.0f}")
    macro("ColourN", str(n))
    print("colour table (2 corruptions):", {c: {k: round(v) for k, v in d.items()} for c, d in res.items()})


def edge(a):  # mean gradient magnitude per image, sketches in [0,1]
    gx = np.abs(np.diff(a, axis=-1))[:, :-1, :]
    gy = np.abs(np.diff(a, axis=-2))[:, :, :-1]
    return np.sqrt(gx ** 2 + gy ** 2).mean((1, 2))


def sharpness_table():
    os.environ.setdefault("FS2K_DIR", str(ROOT / "data/fs2k_raw/FS2K"))
    os.environ.setdefault("CACHE_DIR", str(ROOT / "data/cache"))
    from src.data.fs2k import get_split
    P, S, st = get_split("test")
    sess = ort.InferenceSession(str(ROOT / "onnx_models/task4_generator.onnx"), providers=["CPUExecutionProvider"])

    def gen(photos, styles):
        x = photos.astype(np.float32).transpose(0, 3, 1, 2) / 127.5 - 1
        out = [sess.run(None, {"photo": x[i:i + 64], "style": styles[i:i + 64].astype(np.int64)})[0][:, 0] for i in range(0, len(x), 64)]
        return (np.concatenate(out) + 1) / 2

    gt = S.astype(np.float32) / 255
    g = gen(P, st)
    rows = []
    for s in range(3):
        m = st == s
        er, eg = edge(gt[m]).mean(), edge(g[m]).mean()
        rows.append(f"Style {s + 1} & {int(m.sum())} & {er:.4f} & {eg:.4f} & {100 * eg / er:.0f}\\% \\\\")
        macro(f"SharpShare_{s + 1}".replace("_", ""), f"{100 * eg / er:.0f}")
    er, eg = edge(gt).mean(), edge(g).mean()
    rows += ["\\midrule", f"all & {len(st)} & {er:.4f} & {eg:.4f} & {100 * eg / er:.0f}\\% \\\\"]
    macro("SharpShareAll", f"{100 * eg / er:.0f}")
    rng = np.random.default_rng(0)
    idx = rng.choice(len(P), 300, replace=False)

    def variant(f):
        return np.stack([np.asarray(f(Image.fromarray(p))) for p in P[idx]])

    lines = []
    for name, f in (("input blurred (Gaussian, r=1.5)", lambda im: im.filter(ImageFilter.GaussianBlur(1.5))), ("input as is", lambda im: im),
                    ("input sharpened (unsharp mask)", lambda im: im.filter(ImageFilter.UnsharpMask(radius=1.5, percent=200, threshold=0)))):
        o = gen(variant(f), st[idx])
        lines.append(f"{name} & {edge(o).mean():.4f} & {np.abs(o - gt[idx]).mean():.4f} \\\\")
        macro("Sens" + "".join(ch for ch in name.split("(")[0].title() if ch.isalpha()), f"{edge(o).mean():.4f}")
    (GEN / "tab_sharpness.tex").write_text(
        "\\begin{tabular}{lrccc}\n\\toprule\nTest subset & Pairs & Real sketch edge strength & Generated & Generated / real \\\\\n\\midrule\n" + "\n".join(rows) +
        "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")
    (GEN / "tab_sensitivity.tex").write_text("\\begin{tabular}{lcc}\n\\toprule\nInput photo (300 test faces) & Generated edge strength & L1 to real sketch \\\\\n\\midrule\n" +
                                             "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")
    print("sharpness done")


def latency_table(reps=40):
    """Median single-image CPU latency of each deployed ONNX model (run on an otherwise idle machine)."""
    import time
    out = {}
    x = np.random.default_rng(0).random((1, 3, 128, 128)).astype(np.float32)
    opts = ort.SessionOptions(); opts.intra_op_num_threads = 4
    for f in ("task1_universal_dae.onnx", "task2_classifier.onnx", "task2_specialist_salt_pepper.onnx", "task2_specialist_blur.onnx",
              "task2_specialist_occlusion.onnx", "task3_soft_moe.onnx", "task4_generator.onnx"):
        s = ort.InferenceSession(str(ROOT / "onnx_models" / f), opts, providers=["CPUExecutionProvider"])
        feed = {"photo": x * 2 - 1, "style": np.array([1], np.int64)} if "generator" in f else {"image": x}
        for _ in range(5):
            s.run(None, feed)
        t = []
        for _ in range(reps):
            t0 = time.perf_counter(); s.run(None, feed); t.append((time.perf_counter() - t0) * 1000)
        out[f] = f"{np.median(t):.1f}"
    (GEN / "latency.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("latency", out)


if __name__ == "__main__":
    latency_table()
    colour_table2()
    sharpness_table()
    (GEN / "numbers_analysis.tex").write_text("\n".join(f"\\newcommand{{\\n{k}}}{{{v}}}" for k, v in sorted(MACROS.items())) + "\n", encoding="utf-8")
    print("wrote", len(MACROS), "macros")
