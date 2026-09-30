"""Oxford-IIIT Pet data: split json -> cached uint8 tensors -> corrupted datasets.

Split file format (manifests/pets_split.json): {"train": [names], "val": [names], "test": [names]}
Names are image stems like "Abyssinian_100" (".jpg" is tolerated).
"""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.data.corruptions import IMG_SIZE, TYPE2ID, TYPES, apply_spec, make_spec
from src.utils import paths

_SPLIT = {}
_IMAGES = {}


def load_split(split_json=None) -> dict:
    p = str(split_json or paths.split_json_path())
    if p not in _SPLIT:
        raw = json.load(open(p))
        _SPLIT[p] = {k: [Path(n).stem for n in raw[k]] for k in ("train", "val", "test")}
    return _SPLIT[p]


def load_images_uint8(names, size=IMG_SIZE, cache_path=None) -> torch.Tensor:
    """(N,3,size,size) uint8. RGB, direct resize to size x size (bicubic). Cached as .npy."""
    if cache_path is not None and Path(cache_path).exists():
        return torch.from_numpy(np.load(cache_path))
    img_dir = paths.pets_image_dir()
    out = np.empty((len(names), size, size, 3), dtype=np.uint8)
    for i, n in enumerate(names):
        with Image.open(img_dir / f"{n}.jpg") as im:
            out[i] = np.asarray(im.convert("RGB").resize((size, size), Image.Resampling.BICUBIC))
        if (i + 1) % 1000 == 0:
            print(f"  loaded {i + 1}/{len(names)}")
    out = np.ascontiguousarray(out.transpose(0, 3, 1, 2))
    if cache_path is not None:
        np.save(cache_path, out)
    return torch.from_numpy(out)


def get_images(split: str, size=IMG_SIZE, split_json=None) -> torch.Tensor:
    key = (split, size, str(split_json))
    if key not in _IMAGES:
        names = load_split(split_json)[split]
        cache = paths.cache_dir() / f"pets_{split}_{size}_{len(names)}.npy"
        print(f"[data] loading {split} images ({len(names)})")
        _IMAGES[key] = load_images_uint8(names, size, cache)
    return _IMAGES[key]


# ----------------------------------------------------------------------------- training dataset
class PetsTrainCorrupted(Dataset):
    """Runtime corruption: a NEW spec is sampled every time an item is loaded (equal prob. for the 4 types).

    Item = (corrupted, clean, label, image_index). Accepts either an int index or (index, forced_label),
    the latter is used by BalancedCorruptionBatchSampler for exactly balanced classifier batches.
    """

    def __init__(self, images: torch.Tensor):
        self.images = images
        self._rng, self._rng_seed = None, None

    def _get_rng(self):
        seed = torch.initial_seed()  # differs per worker and per epoch, reproducible under torch.manual_seed
        if self._rng is None or self._rng_seed != seed:
            self._rng, self._rng_seed = np.random.default_rng(seed % (2**32)), seed
        return self._rng

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        idx, forced = item if isinstance(item, tuple) else (int(item), None)
        rng = self._get_rng()
        label = int(forced) if forced is not None else int(rng.integers(0, len(TYPES)))
        spec = make_spec(TYPES[label], rng, "random")
        clean = self.images[idx].float() / 255.0
        return apply_spec(clean, spec), clean, label, idx


class BalancedCorruptionBatchSampler:
    """Every batch has EXACTLY batch_size/4 items of each class (clean/salt/blur/occlusion)."""

    def __init__(self, n_items, batch_size, seed=42):
        assert batch_size % 4 == 0, "batch size must be a multiple of 4"
        self.n, self.bs, self.seed, self.epoch = n_items, batch_size, seed, 0

    def __len__(self):
        return self.n // self.bs

    def __iter__(self):
        g = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        perm = g.permutation(self.n)
        for b in range(len(self)):
            labels = np.repeat(np.arange(4), self.bs // 4)
            g.shuffle(labels)
            ids = perm[b * self.bs:(b + 1) * self.bs]
            yield [(int(i), int(l)) for i, l in zip(ids, labels)]


# ----------------------------------------------------------------------------- manifest replay dataset
class PetsManifestDataset(Dataset):
    """Deterministic replay of a val/test manifest. Item = (corrupted, clean, label, entry_index).
    Per-entry metadata (severity, params, ...) is available via self.entries[entry_index]."""

    def __init__(self, entries, images: torch.Tensor):
        self.entries, self.images = entries, images

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        e = self.entries[i]
        clean = self.images[e["img_index"]].float() / 255.0
        return apply_spec(clean, e), clean, e["label"], i


def load_manifest(path):
    d = json.load(open(path))
    return d["meta"], d["entries"]


# ----------------------------------------------------------------------------- loader builders
def _workers(n):
    return n if n is not None else (2 if paths.is_kaggle() else 0)


def build_train_loader(kind="ae", batch_size=32, max_images=None, num_workers=None, seed=42, split_json=None):
    """kind='ae'  : random corruption type per item, shuffled.
       kind='cls' : exactly balanced batches (batch_size rounded up to a multiple of 4)."""
    images = get_images("train", split_json=split_json)
    if max_images:
        images = images[:max_images]  # split was seeded-random, so a prefix is a random subset
    ds = PetsTrainCorrupted(images)
    nw, pin = _workers(num_workers), torch.cuda.is_available()
    if kind == "cls":
        bs = int(np.ceil(batch_size / 4) * 4)
        return DataLoader(ds, batch_sampler=BalancedCorruptionBatchSampler(len(ds), bs, seed),
                          num_workers=nw, pin_memory=pin)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=nw, pin_memory=pin)


def build_eval_loader(split="val", manifest_path=None, batch_size=64, max_items=None, num_workers=None,
                      split_json=None):
    manifest_path = manifest_path or (paths.manifest_dir() / f"{split}_manifest.json")
    _, entries = load_manifest(manifest_path)
    if max_items:
        entries = entries[:max_items]
    ds = PetsManifestDataset(entries, get_images(split, split_json=split_json))
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=_workers(num_workers),
                      pin_memory=torch.cuda.is_available())


# ----------------------------------------------------------------------------- single-corruption-type loaders (Task 2 specialists)
class _ForcedLabelWrap(Dataset):
    """Wraps PetsTrainCorrupted so every item is forced to one corruption type (used to train a specialist AE)."""

    def __init__(self, base: PetsTrainCorrupted, label: int):
        self.base, self.label = base, label

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        return self.base[(int(i), self.label)]


def build_single_type_loader(ctype, batch_size=32, max_images=None, num_workers=None, shuffle=True, split_json=None):
    """Training loader that always corrupts with `ctype` (e.g. 'salt_pepper'). Used for Task 2 specialists."""
    images = get_images("train", split_json=split_json)
    if max_images:
        images = images[:max_images]
    ds = _ForcedLabelWrap(PetsTrainCorrupted(images), TYPE2ID[ctype])
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True, num_workers=_workers(num_workers),
                      pin_memory=torch.cuda.is_available())


def build_eval_loader_by_type(split, ctype, manifest_path=None, batch_size=64, num_workers=None, split_json=None):
    """Manifest-replay eval loader filtered to a single corruption type (Task 2 specialist validation/test)."""
    manifest_path = manifest_path or (paths.manifest_dir() / f"{split}_manifest.json")
    _, entries = load_manifest(manifest_path)
    entries = [e for e in entries if e["type"] == ctype]
    ds = PetsManifestDataset(entries, get_images(split, split_json=split_json))
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=_workers(num_workers),
                      pin_memory=torch.cuda.is_available())
