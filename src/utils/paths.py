"""Central place for all filesystem paths. Same code runs on Kaggle and locally.

Override with environment variables:
  PETS_IMAGE_DIR : folder that directly contains Abyssinian_1.jpg, ... (the Oxford-IIIT Pet images)
  SPLIT_JSON     : path to pets_split.json      (default: <repo>/manifests/pets_split.json)
  MANIFEST_DIR   : where val/test manifests live (default: <repo>/manifests)
  CACHE_DIR      : where resized uint8 .npy caches go
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def is_kaggle() -> bool:
    return Path("/kaggle/input").exists()


def split_json_path() -> Path:
    return Path(os.environ.get("SPLIT_JSON", REPO_ROOT / "manifests" / "pets_split.json"))


def manifest_dir() -> Path:
    return Path(os.environ.get("MANIFEST_DIR", REPO_ROOT / "manifests"))


def cache_dir() -> Path:
    default = Path("/kaggle/working/cache") if is_kaggle() else REPO_ROOT / "data" / "cache"
    p = Path(os.environ.get("CACHE_DIR", default))
    p.mkdir(parents=True, exist_ok=True)
    return p


def checkpoint_dir() -> Path:
    p = REPO_ROOT / "checkpoints"
    p.mkdir(parents=True, exist_ok=True)
    return p


def pets_image_dir() -> Path:
    env = os.environ.get("PETS_IMAGE_DIR")
    if env:
        return Path(env)
    if is_kaggle():
        # Kaggle copies of Oxford Pets differ in nesting (images/, images/images/, ...): find a known file.
        for dirpath, _, files in os.walk("/kaggle/input"):
            if "Abyssinian_1.jpg" in files:
                return Path(dirpath)
    raise FileNotFoundError(
        "Could not locate the Oxford Pet images. Set PETS_IMAGE_DIR to the folder containing Abyssinian_1.jpg."
    )
