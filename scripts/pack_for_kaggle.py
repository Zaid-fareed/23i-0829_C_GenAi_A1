"""Zip only the training code (no .venv/data/checkpoints/mlruns) into kaggle_code.zip with Linux-style paths.

  python scripts/pack_for_kaggle.py
"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ITEMS = ["src", "scripts", "tests", "configs", "requirements.txt"]

with zipfile.ZipFile(ROOT / "kaggle_code.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for item in ITEMS:
        p = ROOT / item
        files = [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file()] if p.exists() else []
        for f in files:
            if "__pycache__" not in f.parts:
                z.write(f, f.relative_to(ROOT).as_posix())
print("wrote kaggle_code.zip", (ROOT / "kaggle_code.zip").stat().st_size, "bytes")
