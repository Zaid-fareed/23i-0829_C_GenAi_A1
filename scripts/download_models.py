"""Download the seven trained ONNX models from the GitHub release assets into ./onnx_models and verify their checksums.

  python scripts/download_models.py                 # download what is missing or corrupted
  python scripts/download_models.py --force         # re-download everything
  python scripts/download_models.py --tag models-v1 --repo Zaid-fareed/23i-0829_C_GenAi_A1

Needs only the Python standard library. The checksums are listed in onnx_models/SHA256SUMS.txt (committed to Git).
"""
import argparse
import hashlib
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Zaid-fareed/23i-0829_C_GenAi_A1")
ap.add_argument("--tag", default="models-v1", help="release tag that holds the model files")
ap.add_argument("--force", action="store_true")
ap.add_argument("--out", default=str(ROOT / "onnx_models"))
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)

sums = {}
for line in (ROOT / "onnx_models" / "SHA256SUMS.txt").read_text().splitlines():
    if line.strip() and not line.startswith("#"):
        digest, name = line.split()
        sums[name.lstrip("*")] = digest


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


failed = []
for name, digest in sums.items():
    dst = out / name
    if dst.exists() and not a.force and sha256(dst) == digest:
        print(f"ok        {name} (already present, checksum matches)")
        continue
    url = f"https://github.com/{a.repo}/releases/download/{a.tag}/{name}"
    print(f"download  {name} <- {url}")
    try:
        urllib.request.urlretrieve(url, dst)
    except urllib.error.HTTPError as e:
        print(f"  FAILED: HTTP {e.code}. Is the release '{a.tag}' published, the repository public, and the file attached?")
        failed.append(name)
        continue
    if sha256(dst) != digest:
        print("  FAILED: checksum mismatch (corrupted or different file)")
        failed.append(name)
    else:
        print(f"  verified ({dst.stat().st_size / 1e6:.1f} MB)")
if failed:
    print("\nincomplete:", ", ".join(failed))
    sys.exit(1)
print(f"\nall {len(sums)} models are in {out}")
