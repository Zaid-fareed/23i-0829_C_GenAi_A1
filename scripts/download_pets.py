"""Download Oxford-IIIT Pet and verify the required 80/20 split (seed 42)."""
import torch
from torch.utils.data import random_split
from torchvision.datasets import OxfordIIITPet

ROOT = "data"
trainval = OxfordIIITPet(root=ROOT, split="trainval", download=True)
test = OxfordIIITPet(root=ROOT, split="test", download=True)
print(f"official trainval: {len(trainval)} | official test: {len(test)}")

n_val = int(round(0.2 * len(trainval)))
n_train = len(trainval) - n_val
gen = torch.Generator().manual_seed(42)
train_set, val_set = random_split(range(len(trainval)), [n_train, n_val], generator=gen)
print(f"train: {len(train_set)} | val: {len(val_set)}")

# Save the split indices so every task reuses the exact same split
import json, os
os.makedirs("manifests", exist_ok=True)
json.dump({"seed": 42, "train": list(train_set.indices), "val": list(val_set.indices)},
          open("manifests/pets_split.json", "w"))
print("saved manifests/pets_split.json")
