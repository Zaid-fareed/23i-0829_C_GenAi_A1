"""Corruptions for 128x128 RGB float tensors in [0,1], shape (3,H,W).

Design (easy to explain in the viva):
  make_spec(type, rng, severity)  -> a plain dict ("spec") that FULLY describes one corruption
                                     (all randomness is resolved here: p, noise seed, kernel, sigma, rectangle coords).
  apply_spec(img, spec)           -> pure, deterministic function of (img, spec).

Runtime training  : spec is sampled per image load with a fresh RNG.
Val/test manifests: specs are sampled ONCE (fixed seeds), stored in JSON, and replayed with apply_spec.
So training and evaluation share exactly the same corruption code.
"""
import numpy as np
import torch
import torch.nn.functional as F

IMG_SIZE = 128
TYPES = ["clean", "salt_pepper", "blur", "occlusion"]  # class ids 0..3 (same order as MoE branches)
TYPE2ID = {t: i for i, t in enumerate(TYPES)}
SEV_NAMES = ["low", "medium", "high"]

# Fixed test severities (from the assignment)
FIXED = {
    "salt_pepper": {"low": {"p": 0.03}, "medium": {"p": 0.08}, "high": {"p": 0.15}},
    "blur": {
        "low": {"kernel": 3, "sigma": 0.7},
        "medium": {"kernel": 5, "sigma": 1.5},
        "high": {"kernel": 7, "sigma": 2.5},
    },
    "occlusion": {
        "low": {"target": 0.10, "n_rects": 1},
        "medium": {"target": 0.20, "n_rects": 2},
        "high": {"target": 0.35, "n_rects": 3},
    },
}
# Random (training / validation) ranges
SP_P = (0.02, 0.15)
BLUR_KERNELS = (3, 5, 7)
BLUR_SIGMA = (0.5, 2.5)
OCC_AREA = (0.10, 0.35)
OCC_RECTS = (1, 3)  # inclusive
OCC_TOL = 0.01  # a fixed-severity mask must land within +-1% of its target area


# ----------------------------------------------------------------------------- occlusion geometry
def rects_to_mask(rects, size=IMG_SIZE) -> np.ndarray:
    """rects: list of [x0,y0,x1,y1] (x1,y1 exclusive) -> boolean mask (size,size), True = occluded."""
    m = np.zeros((size, size), dtype=bool)
    for x0, y0, x1, y1 in rects:
        m[y0:y1, x0:x1] = True
    return m


def sample_rects(rng, n, target, lo, hi, size=IMG_SIZE, tol=OCC_TOL, max_tries=200):
    """Sample n rectangles whose UNION covers ~target of the image, accepted only if lo<=coverage<=hi.

    Each rectangle gets a share of the area (Dirichlet), an aspect ratio in [0.5,2] and a random position.
    Overlaps shrink the union, so the common scale of all rectangles is corrected iteratively.
    """
    N = size * size
    best = None
    for _ in range(max_tries):
        share = rng.dirichlet(np.full(n, 4.0))
        ar = np.exp(rng.uniform(np.log(0.5), np.log(2.0), n))  # width / height
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


# ----------------------------------------------------------------------------- spec sampling
def make_spec(ctype: str, rng: np.random.Generator, severity: str = "random", size: int = IMG_SIZE) -> dict:
    """severity: 'random' (train/val ranges) or 'low'|'medium'|'high' (fixed test values)."""
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


# ----------------------------------------------------------------------------- appliers
def salt_and_pepper(img: torch.Tensor, p: float, noise_seed: int) -> torch.Tensor:
    """Corrupt exactly round(p*H*W) pixels (all channels of a pixel), each black or white with prob 0.5."""
    _, H, W = img.shape
    g = np.random.default_rng(noise_seed)
    n = int(round(p * H * W))
    pos = g.permutation(H * W)[:n]
    vals = torch.from_numpy((g.random(n) < 0.5).astype(np.float32))  # 1 = salt (white), 0 = pepper (black)
    out = img.clone()
    out[:, torch.from_numpy(pos // W), torch.from_numpy(pos % W)] = vals
    return out


def gaussian_kernel1d(k: int, sigma: float) -> torch.Tensor:
    x = torch.arange(k, dtype=torch.float32) - (k - 1) / 2
    g = torch.exp(-(x**2) / (2 * sigma**2))
    return g / g.sum()


def gaussian_blur(img: torch.Tensor, k: int, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur, reflect padding (own implementation -> no torchvision dependency)."""
    g = gaussian_kernel1d(k, sigma).to(img)
    c, pad = img.shape[0], k // 2
    x = F.pad(img.unsqueeze(0), (pad, pad, pad, pad), mode="reflect")
    x = F.conv2d(x, g.view(1, 1, 1, k).repeat(c, 1, 1, 1), groups=c)
    x = F.conv2d(x, g.view(1, 1, k, 1).repeat(c, 1, 1, 1), groups=c)
    return x.squeeze(0)


def occlude(img: torch.Tensor, rects) -> torch.Tensor:
    out = img.clone()
    out[:, torch.from_numpy(rects_to_mask(rects, img.shape[-1]))] = 0.0
    return out


def apply_spec(img: torch.Tensor, spec: dict) -> torch.Tensor:
    t = spec["type"]
    if t == "clean":
        return img.clone()
    if t == "salt_pepper":
        return salt_and_pepper(img, spec["p"], spec["noise_seed"])
    if t == "blur":
        return gaussian_blur(img, spec["kernel"], spec["sigma"])
    if t == "occlusion":
        return occlude(img, spec["rects"])
    raise ValueError(t)
