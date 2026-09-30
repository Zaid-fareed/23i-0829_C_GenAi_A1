"""Generate the deterministic validation and test corruption manifests (run ONCE, commit the output).

  python -m src.data.make_manifests                # val + test (test_mode=cycle)
  python -m src.data.make_manifests --test_mode full

val  : each image gets one corruption; type assignment is exactly balanced (25% each),
       parameters drawn from the training ranges ('random' severity).
test : 12 conditions = clean x3 slots + {salt, blur, occlusion} x {low, medium, high}  (=> classes 25% each)
         cycle: each image gets ONE condition (balanced, shuffled)   [default]
         full : each image gets ALL 10 conditions (clean once) -> paired comparison, 10x bigger.
Every entry stores: type, severity, params (p/noise_seed | kernel/sigma | rects/coverage), seed, label, image, img_index.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from src.data.corruptions import FIXED, SEV_NAMES, TYPE2ID, TYPES, make_spec
from src.data.pets import load_split
from src.utils import paths

SPLIT_CODE = {"val": 1, "test": 2}
CORRUPT = ["salt_pepper", "blur", "occlusion"]
TEST_CONDITIONS = [("clean", "none")] + [(t, s) for t in CORRUPT for s in SEV_NAMES]  # 10


def _entry(k, names, img_index, ctype, sev, seed, split):
    rng = np.random.default_rng(seed)
    spec = make_spec(ctype, rng, sev)
    return {"idx": k, "split": split, "image": names[img_index], "img_index": img_index,
            "label": TYPE2ID[ctype], "seed": seed, **spec}


def build_entries(names, split, seed=42, test_mode="cycle"):
    code = SPLIT_CODE[split]
    assign_rng = np.random.default_rng(seed + code)
    n, entries = len(names), []
    if split == "val":
        labels = np.tile(np.arange(4), n // 4 + 1)[:n]
        assign_rng.shuffle(labels)
        for i in range(n):
            entries.append(_entry(i, names, i, TYPES[labels[i]], "random", seed * 10_000_000 + code * 1_000_000 + i, split))
    elif test_mode == "cycle":
        slots = [("clean", "none")] * 3 + TEST_CONDITIONS[1:]  # 12 slots
        pick = np.tile(np.arange(12), n // 12 + 1)[:n]
        assign_rng.shuffle(pick)
        for i in range(n):
            c, s = slots[pick[i]]
            entries.append(_entry(i, names, i, c, s, seed * 10_000_000 + code * 1_000_000 + i, split))
    elif test_mode == "full":
        k = 0
        for i in range(n):
            for c, s in TEST_CONDITIONS:
                entries.append(_entry(k, names, i, c, s, seed * 10_000_000 + code * 1_000_000 + k, split))
                k += 1
    else:
        raise ValueError(test_mode)
    return entries


def _digest(entries):
    return hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()[:16]


def write(entries, split, out_dir, seed, test_mode):
    meta = {"split": split, "seed": seed, "mode": test_mode if split == "test" else "balanced_random",
            "n": len(entries), "sha256_16": _digest(entries), "fixed_test_severities": FIXED}
    out_dir.mkdir(parents=True, exist_ok=True)
    jp = out_dir / f"{split}_manifest.json"
    json.dump({"meta": meta, "entries": entries}, open(jp, "w"))
    cols = ["idx", "image", "img_index", "label", "type", "severity", "seed", "p", "noise_seed", "kernel",
            "sigma", "n_rects", "target", "coverage", "rects"]
    with open(out_dir / f"{split}_manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for e in entries:
            w.writerow({**e, "rects": json.dumps(e.get("rects", ""))})
    counts = {}
    for e in entries:
        counts[(e["type"], e["severity"])] = counts.get((e["type"], e["severity"]), 0) + 1
    print(f"[{split}] {len(entries)} entries, digest {meta['sha256_16']} -> {jp}")
    for k in sorted(counts):
        print(f"    {k[0]:12s} {k[1]:8s} {counts[k]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_mode", choices=["cycle", "full"], default="cycle")
    ap.add_argument("--out_dir", default=None)
    a = ap.parse_args()
    out = Path(a.out_dir) if a.out_dir else paths.manifest_dir()
    split = load_split()
    print({k: len(v) for k, v in split.items()})
    assert set(split["train"]).isdisjoint(split["val"]) and set(split["train"] + split["val"]).isdisjoint(split["test"])
    write(build_entries(split["val"], "val", a.seed), "val", out, a.seed, a.test_mode)
    write(build_entries(split["test"], "test", a.seed, a.test_mode), "test", out, a.seed, a.test_mode)


if __name__ == "__main__":
    main()
