"""Fill backend/samples/ with real demo images (resized to <=384px, small JPEGs):
   pet_*.jpg  : Oxford-IIIT Pet images from the OFFICIAL TEST split (never used for training), different breeds
   face_*.jpg : FS2K official-TEST photographs, covering the three sketch styles

  python scripts/make_samples.py --pets data/oxford-iiit-pet/images --fs2k data/fs2k_raw/FS2K
"""
import argparse
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--pets", default=str(ROOT / "data/oxford-iiit-pet/images"))
ap.add_argument("--fs2k", default=str(ROOT / "data/fs2k_raw/FS2K"))
a = ap.parse_args()
out = ROOT / "backend/samples"
out.mkdir(parents=True, exist_ok=True)
for old in out.glob("demo_*.jpg"):
    old.unlink()


def save(src, dst):
    with Image.open(src) as im:
        im = im.convert("RGB")
        im.thumbnail((384, 384))
        im.save(out / dst, quality=88)


# --- pets: one image per breed from the test split, spread over breeds
test = json.load(open(ROOT / "manifests/pets_split.json"))["test"]
seen, picked = set(), []
for n in sorted(test):
    breed = n.rsplit("_", 1)[0]
    if breed not in seen:
        seen.add(breed); picked.append(n)
step = max(1, len(picked) // 6)
for n in picked[::step][:6]:
    save(Path(a.pets) / f"{n}.jpg", f"pet_{n}.jpg")

# --- faces: 2 style-1, 1 style-2, 1 style-3 photos from the official FS2K test annotations
anno = json.load(open(Path(a.fs2k) / "anno_test.json"))
want = {0: 2, 1: 1, 2: 1}
for r in anno:
    s = int(r.get("style", r.get("sketch_style")))
    if want.get(s, 0) > 0:
        want[s] -= 1
        stem = r["image_name"]
        src = next(p for ext in (".jpg", ".png", ".jpeg") if (p := Path(a.fs2k) / "photo" / f"{stem}{ext}").exists())
        save(src, f"face_{Path(stem).name}_style{s + 1}.jpg")
print(sorted(p.name for p in out.glob("*.jpg")))
