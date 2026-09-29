"""Create manifests/pets_split.json: official trainval -> 80/20 train/val (seed 42), official test untouched.

Works on Kaggle (needs Internet ON) and locally. Downloads Oxford-IIIT Pet via torchvision, then writes image
NAMES (not indices) because src/data/pets.py loads images by name. Prints the PETS_IMAGE_DIR to export.

  python scripts/make_pets_split.py
"""
import json
import os
from pathlib import Path

import torch
from torch.utils.data import random_split
from torchvision.datasets import OxfordIIITPet

root = Path("/kaggle/working/data" if Path("/kaggle/working").exists() else "data")
trainval = OxfordIIITPet(root=str(root), split="trainval", download=True)
test = OxfordIIITPet(root=str(root), split="test", download=True)
stems = lambda ds: [Path(p).stem for p in ds._images]
tv, te = stems(trainval), stems(test)

n_val = int(round(0.2 * len(tv)))
tr_idx, va_idx = random_split(range(len(tv)), [len(tv) - n_val, n_val], generator=torch.Generator().manual_seed(42))
split = {"seed": 42, "train": [tv[i] for i in tr_idx.indices], "val": [tv[i] for i in va_idx.indices], "test": te}
assert set(split["train"]).isdisjoint(split["val"]) and set(tv).isdisjoint(te)
print({k: len(v) for k, v in split.items() if k != "seed"})

os.makedirs("manifests", exist_ok=True)
json.dump(split, open("manifests/pets_split.json", "w"))
print("saved manifests/pets_split.json")
print("export PETS_IMAGE_DIR=" + str((root / "oxford-iiit-pet" / "images").resolve()))
