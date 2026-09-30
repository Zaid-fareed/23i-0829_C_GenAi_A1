"""FS2K paired photo/sketch data (Task 4).

Layout (official): <root>/photo/photoN/imageXXXX.jpg, <root>/sketch/sketchN/sketchXXXX.jpg,
                   <root>/anno_train.json, <root>/anno_test.json  (each entry has "image_name" and "style" in {0,1,2}).
Root is auto-detected on Kaggle (folder containing anno_train.json) or set with FS2K_DIR.

Split: official train -> 85% train / 15% val stratified by style (seed 42); official test untouched.
Photos -> RGB 128x128 ; sketches -> grayscale 128x128 ; both scaled to [-1, 1] for the tanh generator.
Spatial augmentation (flip + random crop-resize) uses ONE random draw applied to BOTH images of a pair.
"""
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from src.utils import paths

SIZE = 128
STYLES = ["Style 1", "Style 2", "Style 3"]
EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".PNG")
_CACHE = {}


def fs2k_root() -> Path:
    env = os.environ.get("FS2K_DIR")
    if env:
        return Path(env)
    for base in ("/kaggle/input", str(paths.REPO_ROOT / "data")):
        for dirpath, _, files in os.walk(base):
            if "anno_train.json" in files:
                return Path(dirpath)
    raise FileNotFoundError("FS2K not found. Set FS2K_DIR to the folder containing anno_train.json.")


def _find(base: Path, stem: str):
    for e in EXTS:
        if (base / (stem + e)).exists():
            return base / (stem + e)
    return None


def _resolve_pair(root: Path, name: str):
    """name like 'photo1/image0110' -> (photo_path, sketch_path). Tries several sketch naming conventions."""
    photo = _find(root / "photo", name)
    if photo is None:
        raise FileNotFoundError(f"photo for {name} not found under {root / 'photo'}")
    folder, stem = name.split("/")[-2:] if "/" in name else ("", name)
    num = re.sub(r"\D", "", stem)
    cands = [f"{folder.replace('photo', 'sketch')}/{stem.replace('image', 'sketch')}",
             f"{folder.replace('photo', 'sketch')}/{stem}"]
    for c in cands:
        s = _find(root / "sketch", c)
        if s:
            return photo, s
    for d in sorted((root / "sketch").iterdir()):  # fallback: same number in any sketch folder
        if d.is_dir():
            s = _find(d, f"sketch{num}") or _find(d, f"image{num}")
            if s:
                return photo, s
    raise FileNotFoundError(f"sketch for {name} not found (tried {cands} and number match {num})")


def _load_anno(root: Path, split: str):
    raw = json.load(open(root / f"anno_{split}.json"))
    if isinstance(raw, dict):
        raw = list(raw.values())
    return [{"name": r["image_name"], "style": int(r.get("style", r.get("sketch_style")))} for r in raw]


def _load_arrays(root, records, cache_file):
    if cache_file.exists():
        z = np.load(cache_file)
        return z["photo"], z["sketch"], z["style"]
    P = np.empty((len(records), SIZE, SIZE, 3), np.uint8)
    S = np.empty((len(records), SIZE, SIZE), np.uint8)
    for i, r in enumerate(records):
        pp, sp = _resolve_pair(root, r["name"])
        with Image.open(pp) as im:
            P[i] = np.asarray(im.convert("RGB").resize((SIZE, SIZE), Image.Resampling.BICUBIC))
        with Image.open(sp) as im:
            S[i] = np.asarray(im.convert("L").resize((SIZE, SIZE), Image.Resampling.BICUBIC))
    st = np.array([r["style"] for r in records], np.int64)
    np.savez_compressed(cache_file, photo=P, sketch=S, style=st)
    return P, S, st


def get_split(split: str):
    """split in {'train','val','test'} -> (photos uint8 NHWC, sketches uint8 NHW, styles int64)."""
    if split in _CACHE:
        return _CACHE[split]
    root = fs2k_root()
    off = "test" if split == "test" else "train"
    recs = _load_anno(root, off)
    if split != "test":
        idx = np.arange(len(recs))
        tr, va = train_test_split(idx, test_size=0.15, random_state=42, stratify=[r["style"] for r in recs])
        recs = [recs[i] for i in (tr if split == "train" else va)]
    print(f"[fs2k] {split}: {len(recs)} pairs, style counts {np.bincount([r['style'] for r in recs], minlength=3)}")
    _CACHE[split] = _load_arrays(root, recs, paths.cache_dir() / f"fs2k_{split}_{SIZE}.npz")
    return _CACHE[split]


class FS2KDataset(Dataset):
    def __init__(self, split, augment=False, max_items=None, photometric=False):
        p, s, st = get_split(split)
        n = max_items or len(p)
        self.p, self.s, self.st, self.augment = p[:n], s[:n], st[:n], augment
        self.photometric = photometric  # brightness/contrast jitter on the PHOTO only (not spatial -> pairing intact)

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        x = torch.from_numpy(self.p[i]).permute(2, 0, 1).float() / 127.5 - 1
        y = torch.from_numpy(self.s[i])[None].float() / 127.5 - 1
        if self.augment:  # SAME random parameters for photo and sketch (keeps pixel correspondence)
            if torch.rand(1) < 0.5:
                x, y = x.flip(-1), y.flip(-1)
            c = int(torch.randint(int(SIZE * 0.85), SIZE + 1, (1,)))
            t, l = (int(v) for v in torch.randint(0, SIZE - c + 1, (2,)))
            x = F.interpolate(x[None, :, t:t + c, l:l + c], size=SIZE, mode="bilinear", align_corners=False)[0]
            y = F.interpolate(y[None, :, t:t + c, l:l + c], size=SIZE, mode="bilinear", align_corners=False)[0]
        if self.augment and self.photometric:
            x01 = (x + 1) / 2
            b, c = (float(v) for v in torch.empty(2).uniform_(0.85, 1.15))
            x = (((x01 - x01.mean()) * c + x01.mean()) * b).clamp(0, 1) * 2 - 1
        return x, y, int(self.st[i])


def build_loader(split, batch_size=16, augment=None, max_items=None, shuffle=None, workers=None, photometric=False):
    train = split == "train"
    ds = FS2KDataset(split, augment=train if augment is None else augment, max_items=max_items,
                     photometric=photometric and train)
    nw = workers if workers is not None else (2 if paths.is_kaggle() else 0)
    return DataLoader(ds, batch_size=batch_size, shuffle=train if shuffle is None else shuffle, drop_last=train and len(ds) > batch_size,
                      num_workers=nw, pin_memory=torch.cuda.is_available())
