"""NumPy port of src/data/corruptions.py so the backend image does not need PyTorch.

Images are float32 arrays (3,H,W) in [0,1]. `make_spec` / `sample_rects` are identical to the training code;
the appliers are numpy re-implementations verified against the torch versions by backend/tests/test_corruptions.py.
"""
import numpy as np

IMG_SIZE = 128
TYPES = ["clean", "salt_pepper", "blur", "occlusion"]

FIXED = {
    "salt_pepper": {"low": {"p": 0.03}, "medium": {"p": 0.08}, "high": {"p": 0.15}},
    "blur": {"low": {"kernel": 3, "sigma": 0.7}, "medium": {"kernel": 5, "sigma": 1.5},
             "high": {"kernel": 7, "sigma": 2.5}},
    "occlusion": {"low": {"target": 0.10, "n_rects": 1}, "medium": {"target": 0.20, "n_rects": 2},
                  "high": {"target": 0.35, "n_rects": 3}},
}
SP_P, BLUR_KERNELS, BLUR_SIGMA, OCC_AREA, OCC_RECTS, OCC_TOL = (0.02, 0.15), (3, 5, 7), (0.5, 2.5), (0.10, 0.35), (1, 3), 0.01


def rects_to_mask(rects, size=IMG_SIZE):
    m = np.zeros((size, size), dtype=bool)
    for x0, y0, x1, y1 in rects:
        m[y0:y1, x0:x1] = True
    return m


def sample_rects(rng, n, target, lo, hi, size=IMG_SIZE, tol=OCC_TOL, max_tries=200):
    N = size * size
    best = None
    for _ in range(max_tries):
        share = rng.dirichlet(np.full(n, 4.0))
        ar = np.exp(rng.uniform(np.log(0.5), np.log(2.0), n))
        u = rng.random((n, 2))
        scale = target * N
        for _ in range(8):
            rects = []
            for i in range(n):
                h = int(round(np.sqrt(share[i] * scale / ar[i])))
                w = int(round(h * ar[i]))
                h, w = min(max(h, 2), size), min(max(w, 2), size)
                x0 = int(round(u[i, 0] * (size - w)))
                y0 = int(round(u[i, 1] * (size - h)))
                rects.append([x0, y0, x0 + w, y0 + h])
            cov = float(rects_to_mask(rects, size).mean())
            if abs(cov - target) <= tol * 0.5:
                break
            scale *= target / max(cov, 1e-3)
        score = abs(cov - target) + (0.0 if lo <= cov <= hi else 1.0)
        if best is None or score < best[0]:
            best = (score, rects, cov)
        if abs(cov - target) <= tol and lo <= cov <= hi:
            break
    return best[1], best[2]


def make_spec(ctype, rng, severity="random", size=IMG_SIZE):
    if ctype == "clean":
        return {"type": "clean", "severity": "none"}
    spec = {"type": ctype, "severity": severity}
    if ctype == "salt_pepper":
        p = float(rng.uniform(*SP_P)) if severity == "random" else FIXED[ctype][severity]["p"]
        spec.update(p=p, noise_seed=int(rng.integers(0, 2**31 - 1)))
    elif ctype == "blur":
        if severity == "random":
            k, s = int(rng.choice(BLUR_KERNELS)), float(rng.uniform(*BLUR_SIGMA))
        else:
            k, s = FIXED[ctype][severity]["kernel"], FIXED[ctype][severity]["sigma"]
        spec.update(kernel=k, sigma=s)
    elif ctype == "occlusion":
        if severity == "random":
            n = int(rng.integers(OCC_RECTS[0], OCC_RECTS[1] + 1))
            target = float(rng.uniform(*OCC_AREA))
            lo, hi = OCC_AREA
        else:
            n, target = FIXED[ctype][severity]["n_rects"], FIXED[ctype][severity]["target"]
            lo, hi = target - OCC_TOL, target + OCC_TOL
        rects, cov = sample_rects(rng, n, target, lo, hi, size)
        spec.update(n_rects=n, target=target, coverage=cov, rects=rects)
    else:
        raise ValueError(ctype)
    return spec


def custom_spec(ctype, seed, *, sp_p=0.08, blur_kernel=5, blur_sigma=1.5, occ_coverage=0.2, occ_rects=2):
    """User-chosen ('custom') settings from the UI sliders; ranges are validated by the API layer."""
    rng = np.random.default_rng(seed)
    if ctype == "salt_pepper":
        return {"type": ctype, "severity": "custom", "p": float(sp_p), "noise_seed": int(rng.integers(0, 2**31 - 1))}
    if ctype == "blur":
        return {"type": ctype, "severity": "custom", "kernel": int(blur_kernel), "sigma": float(blur_sigma)}
    if ctype == "occlusion":
        rects, cov = sample_rects(rng, int(occ_rects), float(occ_coverage), occ_coverage - OCC_TOL, occ_coverage + OCC_TOL)
        return {"type": ctype, "severity": "custom", "n_rects": int(occ_rects), "target": float(occ_coverage),
                "coverage": cov, "rects": rects}
    raise ValueError(ctype)


def _gauss1d(k, sigma):
    x = np.arange(k, dtype=np.float32) - (k - 1) / 2
    g = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return (g / g.sum()).astype(np.float32)


def apply_spec(img, spec):
    t = spec["type"]
    out = img.copy()
    if t == "clean":
        return out
    _, H, W = img.shape
    if t == "salt_pepper":
        g = np.random.default_rng(spec["noise_seed"])
        n = int(round(spec["p"] * H * W))
        pos = g.permutation(H * W)[:n]
        vals = (g.random(n) < 0.5).astype(np.float32)  # 1 = salt (white), 0 = pepper (black)
        out[:, pos // W, pos % W] = vals
        return out
    if t == "blur":
        k = spec["kernel"]
        g, pad = _gauss1d(k, spec["sigma"]), k // 2
        x = np.pad(img, ((0, 0), (pad, pad), (pad, pad)), mode="reflect")
        x = sum(g[i] * x[:, :, i:i + W] for i in range(k))          # horizontal pass (width = W after crop)
        x = sum(g[i] * x[:, i:i + H, :] for i in range(k))          # vertical pass
        return x.astype(np.float32)
    if t == "occlusion":
        out[:, rects_to_mask(spec["rects"], W)] = 0.0
        return out
    raise ValueError(t)
