"""Zip the LaTeX project (only the files the report actually uses) into report/i230829_C_Report_latex.zip.

  python report/make_overleaf_zip.py
Upload the zip in Overleaf (New Project -> Upload Project), set main.tex as the main document and compile with pdfLaTeX.
"""
import re
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "i230829_C_Report_latex.zip"

sections = sorted((HERE / "sections").glob("*.tex"))
source = "".join(f.read_text(encoding="utf-8") for f in sections + [HERE / "main.tex"])
used_figures = {Path(m).name for m in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", source)}
used_tables = {Path(m).name for m in re.findall(r"\\input\{generated/([^}]+)\}", source)}

files = [HERE / "main.tex", HERE / "refs.bib"] + sections
files += [f for f in sorted((HERE / "figures").iterdir()) if f.name in used_figures]
files += [HERE / "generated" / n for n in sorted(used_tables)]
missing = [f for f in files if not f.exists()]
if missing:
    raise SystemExit(f"missing files: {missing}")
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for f in files:
        z.write(f, f.relative_to(HERE).as_posix())
print(f"wrote {OUT.name}: {len(files)} files, {OUT.stat().st_size / 1e6:.1f} MB")
