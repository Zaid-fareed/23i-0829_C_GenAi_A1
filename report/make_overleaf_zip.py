"""Zip the LaTeX project (everything needed to compile on Overleaf) into report/Report_Assignment1_latex.zip.

  python report/make_overleaf_zip.py
Upload the zip in Overleaf (New Project -> Upload Project), set main.tex as the main document and compile with pdfLaTeX.
"""
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "Report_Assignment1_latex.zip"
files = [HERE / "main.tex", HERE / "refs.bib"]
for folder in ("sections", "generated", "figures"):
    files += [f for f in (HERE / folder).rglob("*") if f.is_file() and f.name != "latency.json"]
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for f in files:
        z.write(f, f.relative_to(HERE).as_posix())
print(f"wrote {OUT.name}: {len(files)} files, {OUT.stat().st_size / 1e6:.1f} MB")
