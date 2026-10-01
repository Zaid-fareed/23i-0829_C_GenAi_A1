"""Report every generated-number macro (\nName) that the LaTeX sources use but that is not defined in generated/numbers*.tex.

  python report/check_macros.py
"""
import glob
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
defined = set()
for f in ("generated/numbers.tex", "generated/numbers_analysis.tex"):
    defined |= set(re.findall(r"\\newcommand\{\\(n[A-Za-z]+)\}", (HERE / f).read_text(encoding="utf-8")))
used = {}
for f in glob.glob(str(HERE / "sections" / "*.tex")) + [str(HERE / "main.tex")]:
    for m in re.findall(r"\\(n[A-Z][A-Za-z]+)", Path(f).read_text(encoding="utf-8")):
        used.setdefault(m, set()).add(Path(f).name)
missing = {m: sorted(v) for m, v in used.items() if m not in defined}
print(f"{len(defined)} macros defined, {len(used)} used")
if missing:
    for m, v in sorted(missing.items()):
        print("UNDEFINED", "\\" + m, "in", ", ".join(v))
    sys.exit(1)
print("all used macros are defined")
