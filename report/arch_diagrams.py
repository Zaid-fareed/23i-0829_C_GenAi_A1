"""Architecture / system diagrams for the report (drawn with matplotlib; box text auto-shrinks to fit)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

FIG = Path(__file__).resolve().parent / "figures"
C = {"in": "#e2e8f0", "enc": "#c7d2fe", "lat": "#fde68a", "dec": "#bbf7d0", "out": "#e2e8f0", "gate": "#fbcfe8",
     "id": "#f1f5f9", "sp": "#fecaca", "bl": "#bae6fd", "oc": "#bbf7d0", "ed": "#4f46e5"}


def canvas(w, h, xmax, ymax):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text, fc="#e0e7ff", fs=6.3, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.08", fc=fc, ec=C["ed"], lw=0.8))
    t = ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                weight="bold" if bold else "normal", linespacing=1.25)
    r, inv = ax.figure.canvas.get_renderer(), ax.transData.inverted()
    for _ in range(25):  # shrink the text until it fits inside the box
        bb = t.get_window_extent(r)
        (x0, y0), (x1, y1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
        if (x1 - x0) <= 0.92 * w and (y1 - y0) <= 0.9 * h:
            break
        t.set_fontsize(t.get_fontsize() * 0.94)


def arrow(ax, p1, p2, color="#334155", lw=0.9):
    ax.annotate("", xy=p2, xytext=p1, arrowprops=dict(arrowstyle="-|>", lw=lw, color=color, shrinkA=0, shrinkB=0))


def label(ax, x, y, text, fs=6.3, ha="center", color="#334155", style="normal"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fs, color=color, style=style)


def fig_arch_ae():
    fig, ax = canvas(7.1, 1.9, 14.2, 3.8)
    xs = [0.1, 2.9, 6.1, 9.3, 12.1]
    ws = [2.2, 2.7, 2.7, 2.4, 2.0]
    specs = [("Corrupted input $\\tilde{x}$\n3×128×128", C["in"]),
             ("Encoder $E$\n4 × [Conv3×3 s2 + Conv3×3]\nBN + ReLU\n48→96→192→384 ch\n→ 384×8×8", C["enc"]),
             ("Bottleneck\n1×1 conv → 32 ch\nlatent $z$: 32×8×8\n= 2,048 values (4.2%)\nDropout2d", C["lat"]),
             ("Decoder $D$\n4 × [Up×2 + 2 Conv3×3]\nBN + ReLU\n384→…→48 ch\n→ 48×128×128", C["dec"]),
             ("Conv3×3 + Sigmoid\nRestored $\\hat{x}$\n3×128×128", C["out"])]
    for x, w, (t, fc) in zip(xs, ws, specs):
        box(ax, x, 1.2, w, 2.1, t, fc)
    for i in range(4):
        arrow(ax, (xs[i] + ws[i], 2.25), (xs[i + 1], 2.25))
    label(ax, 7.1, 0.55, "$\\mathcal{L}_{UDAE}=\\alpha\\,\\mathcal{L}_{L1}(x,\\hat{x})+(1-\\alpha)\\,(1-\\mathrm{SSIM}(x,\\hat{x})),\\ \\alpha=0.58$  |  "
                         "no skip connections: everything passes through the latent $z$  |  4.0 M parameters", fs=6.3)
    label(ax, 7.1, 3.55, "Task 1 - universal denoising autoencoder (one model for clean, salt-and-pepper, blur and occlusion inputs)",
          fs=7, color="#0f172a")
    fig.savefig(FIG / "arch_task1.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_arch_routing():
    fig, ax = canvas(7.1, 4.0, 14.2, 8.0)
    label(ax, 7.1, 7.75, "Task 2 - hard routing (classifier + specialists)", fs=7, color="#0f172a")
    box(ax, 0.1, 5.3, 1.9, 1.5, "Corrupted\ninput $\\tilde{x}$", C["in"])
    box(ax, 2.7, 5.3, 2.7, 1.5, "Classifier $C$\nconv stem (16→128 ch)\nGAP + Dropout + Linear", C["gate"])
    box(ax, 6.1, 5.3, 2.1, 1.5, "class probabilities $p$\n$r=\\arg\\max_k\\,p_k$", C["lat"])
    arrow(ax, (2.0, 6.05), (2.7, 6.05))
    arrow(ax, (5.4, 6.05), (6.1, 6.05))
    names = [("Identity bypass ($r$ = clean)", C["id"]), ("$A_{salt}$", C["sp"]), ("$A_{blur}$", C["bl"]), ("$A_{occ}$", C["oc"])]
    for (t, fc), y in zip(names, [6.55, 5.85, 5.15, 4.45]):
        box(ax, 9.3, y - 0.27, 2.5, 0.55, t, fc)
        arrow(ax, (8.2, 6.05), (9.3, y))
        arrow(ax, (11.8, y), (12.4, 5.55))
    box(ax, 12.4, 5.05, 1.7, 1.0, "Restored\n$\\hat{x}$", C["out"])
    label(ax, 7.1, 3.85, "Specialists share one architecture (Task 1 autoencoder, 48→384 ch, bottleneck 32) but have independent weights", fs=6)
    label(ax, 7.1, 3.35, "Task 3 - soft mixture of experts (jointly trained)", fs=7, color="#0f172a")
    box(ax, 0.1, 1.5, 1.9, 1.3, "Corrupted\ninput $\\tilde{x}$", C["in"])
    box(ax, 2.7, 1.5, 2.6, 1.3, "Gate $G$\n(initialised from $C$)\nlogits ÷ $\\tau$, softmax", C["gate"])
    box(ax, 5.9, 1.5, 2.2, 1.3, "weights $w$\n(4 values, sum = 1)\n$\\tau=2.68$", C["lat"])
    arrow(ax, (2.0, 2.15), (2.7, 2.15))
    arrow(ax, (5.3, 2.15), (5.9, 2.15))
    names = [("$w_0\\cdot\\tilde{x}$  (identity)", C["id"]), ("$w_1\\cdot A_{salt}$", C["sp"]),
             ("$w_2\\cdot A_{blur}$", C["bl"]), ("$w_3\\cdot A_{occ}$", C["oc"])]
    for (t, fc), y in zip(names, [2.85, 2.25, 1.65, 1.05]):
        box(ax, 9.3, y - 0.26, 2.5, 0.52, t, fc)
        arrow(ax, (8.1, 2.15), (9.3, y))
        arrow(ax, (11.8, y), (12.3, 1.95))
    box(ax, 12.3, 1.35, 1.85, 1.2, "Restored\n$\\hat{x}=\\Sigma_k\\,w_k\\,b_k(\\tilde{x})$", C["out"])
    label(ax, 7.1, 0.4, "Loss: $\\lambda_1\\mathcal{L}_{L1}+\\lambda_s(1-\\mathrm{SSIM})+\\lambda_c\\mathcal{L}_{CE}+\\lambda_b\\mathcal{L}_{balance}$;  "
                        "stage 1: experts frozen, gate only;  stage 2: all unfrozen, smaller lr", fs=6.2)
    fig.savefig(FIG / "arch_task2_task3.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_arch_cgan():
    fig, ax = canvas(7.1, 3.6, 14.2, 7.2)
    label(ax, 7.1, 6.95, "Task 4 - style-conditioned pix2pix-style cGAN", fs=7, color="#0f172a")
    box(ax, 0.1, 4.45, 1.7, 1.5, "Photo $x$\n3×128×128", C["in"])
    box(ax, 0.1, 6.05, 1.7, 0.75, "Style $s\\in\\{1,2,3\\}$\nEmbedding(3, 32)", C["lat"])
    box(ax, 2.3, 4.45, 3.4, 1.5, "Encoder: 7 down-blocks\n[x, emb map] (35 ch)\n64→128→256→512 (×4)\nBN + LeakyReLU", C["enc"])
    box(ax, 6.2, 4.45, 2.2, 1.5, "1×1 bottleneck\n+ emb (32 ch)", C["lat"])
    box(ax, 8.9, 4.45, 3.3, 1.5, "Decoder: 6 up-blocks\n+ encoder skip links\nDropout 0.3 (first 3)", C["dec"])
    box(ax, 12.7, 4.45, 1.4, 1.5, "ConvT + tanh\nSketch $\\hat{y}$\n1×128×128", C["out"])
    arrow(ax, (1.8, 5.2), (2.3, 5.2))
    arrow(ax, (0.95, 6.05), (0.95, 5.95), color="#64748b")
    arrow(ax, (5.7, 5.2), (6.2, 5.2))
    arrow(ax, (8.4, 5.2), (8.9, 5.2))
    arrow(ax, (12.2, 5.2), (12.7, 5.2))
    label(ax, 7.1, 3.95, "Generator $G(x,s)$: U-Net, 42.1 M parameters (PatchGAN $D$: 2.8 M); "
                         "the style embedding enters the input AND the bottleneck", fs=6.2)
    box(ax, 0.1, 1.45, 3.3, 1.6, "Pair (36 ch):\nphoto $x$, sketch $y$ or $\\hat{y}$,\nstyle-embedding map", C["in"])
    box(ax, 3.9, 1.45, 4.4, 1.6, "PatchGAN $D$\nConv4×4 s2: 64→128→256\nConv4×4 s1: 512→1\nBN + LeakyReLU", C["gate"])
    box(ax, 8.8, 1.45, 2.4, 1.6, "14×14 patch logits\nreal → 1, fake → 0\nBCE with logits", C["lat"])
    box(ax, 11.7, 1.45, 2.4, 1.6, "$\\mathcal{L}_G=\\mathcal{L}_{adv}+\\lambda\\,\\mathcal{L}_{L1}$, $\\lambda$=121\n"
                                 "Adam (0.5, 0.999), batch 8\nlr$_G$ 1.5e-4, lr$_D$ 8.6e-4", C["out"])
    arrow(ax, (3.4, 2.25), (3.9, 2.25))
    arrow(ax, (8.3, 2.25), (8.8, 2.25))
    arrow(ax, (11.2, 2.25), (11.7, 2.25))
    label(ax, 7.1, 0.75, "Spatial augmentation (flip, random crop-resize) uses ONE random draw applied to both photo and sketch",
          fs=6.2, style="italic")
    fig.savefig(FIG / "arch_task4.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_arch_system():
    fig, ax = canvas(7.1, 2.5, 14.2, 5.0)
    ax.add_patch(FancyBboxPatch((3.15, 0.25), 10.95, 3.9, boxstyle="round,pad=0,rounding_size=0.12", fc="#f8fafc",
                                ec="#94a3b8", lw=0.8, ls="--"))
    label(ax, 8.6, 3.92, "Docker Compose (single command: docker compose up --build)", fs=6.3, color="#475569", style="italic")
    box(ax, 0.1, 1.4, 2.6, 2.0, "Browser\nReact + Tailwind UI\n(designed in Google Stitch)\nupload / webcam / samples", C["enc"])
    box(ax, 3.5, 1.4, 2.5, 2.0, "frontend container\nnginx :3000\nstatic build +\nproxy /api", C["lat"])
    box(ax, 6.8, 1.4, 3.1, 2.0, "backend container\nFastAPI :8000\nvalidate upload, preprocess,\nruntime corruption, timing", C["dec"])
    box(ax, 10.6, 1.4, 3.3, 2.0, "ONNX Runtime (CPU)\n7 models from ./onnx_models\nuniversal, classifier, 3 specialists,\nsoft MoE, sketch generator", C["gate"])
    for a, b in ((2.7, 3.5), (6.0, 6.8), (9.9, 10.6)):
        arrow(ax, (a, 2.65), (b, 2.65))
        arrow(ax, (b, 2.15), (a, 2.15))
    box(ax, 6.8, 0.4, 3.1, 0.65, "MLflow UI :5000 (optional profile)\nreads mlflow.db", "#e0f2fe")
    label(ax, 1.4, 0.7, "responses: JSON + PNG data-URLs\n(images, routing weights, timings)", fs=5.8, style="italic")
    fig.savefig(FIG / "arch_system.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    for fn in (fig_arch_ae, fig_arch_routing, fig_arch_cgan, fig_arch_system):
        fn()
        print("ok", fn.__name__)
