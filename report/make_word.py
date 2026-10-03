"""Build the Word version of the report from the SAME LaTeX sources (so the two versions cannot disagree).

  python report/make_word.py        -> report/i230829_C_Report.docx

Pipeline: read figure/table/section numbers from the LaTeX build (main.aux) -> copy the sources to report/_word and
resolve references, captions and wide-table wrappers -> convert PDF figures to PNG -> Pandoc (LaTeX to DOCX, IEEE
citation style) -> python-docx post-processing (fonts, margins, table borders, picture sizes).
Requires: pypandoc_binary, python-docx, pymupdf, and a finished LaTeX build (tectonic -X compile main.tex --keep-intermediates).
"""
import re
import shutil
import subprocess
from pathlib import Path

import pymupdf
import pypandoc
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

HERE = Path(__file__).resolve().parent
TMP = HERE / "_word"
OUT = HERE / "i230829_C_Report.docx"
ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}


def roman(s):
    total = 0
    for i, ch in enumerate(s):
        v = ROMAN[ch]
        total += -v if i + 1 < len(s) and ROMAN[s[i + 1]] > v else v
    return total


def read_labels():
    out = {}
    for line in (HERE / "main.aux").read_text(encoding="utf-8").splitlines():
        m = re.match(r"\\newlabel\{([^}]+)\}\{\{(.*?)\}\{\d+\}", line)
        if not m:
            continue
        key, value = m.groups()
        if key.startswith("sec:"):            # Word numbers sections 1, 1.1 ...: take the numeric anchor, not the roman value
            a = re.search(r"\{(?:sub)*section\.([0-9.]+)\}\{\}\}\s*$", line)
            out[key] = a.group(1) if a else value
        else:
            out[key] = value
    return out


def balanced(s, start):
    """s[start] == '{' -> index of the matching '}'."""
    depth = 0
    for i in range(start, len(s)):
        depth += (s[i] == "{") - (s[i] == "}")
        if depth == 0:
            return i
    raise ValueError("unbalanced braces")


def fix_floats(text, labels):
    def repl(m):
        kind, body = m.group(1), m.group(2)
        i = body.find("\\caption{")
        if i < 0:
            return m.group(0)
        j = balanced(body, i + len("\\caption"))
        caption = body[i + len("\\caption{"):j]
        rest = body[j + 1:]
        lab = re.match(r"\s*\\label\{([^}]+)\}", rest)
        num = labels.get(lab.group(1), "?") if lab else "?"
        if lab:
            rest = rest[lab.end():]
        word = "Fig." if kind == "figure" else "Table"
        body = body[:i] + "\\caption{\\textbf{" + word + "~" + num + ".} " + caption + "}" + rest
        return f"\\begin{{{kind}}}{body}\\end{{{kind}}}"
    return re.sub(r"\\begin\{(figure|table)\*?\}(.*?)\\end\{\1\*?\}", repl, text, flags=re.S)


def transform(text, labels, name):
    text = re.sub(r"\\adjustbox\{[^}]*\}\{(\\input\{[^}]*\})\}", r"\1", text)
    text = text.replace("\\begin{figure*}", "\\begin{figure}").replace("\\end{figure*}", "\\end{figure}")
    text = text.replace("\\begin{table*}", "\\begin{table}").replace("\\end{table*}", "\\end{table}")
    text = re.sub(r"\\includegraphics\[width=([^\]]*)\]\{([^}]*?)\.(?:pdf|png)\}", r"\\includegraphics[width=\1]{figures/\2.png}", text)
    text = fix_floats(text, labels)

    def ref(m):
        k = m.group(1)
        if k not in labels:
            print("  WARNING: unresolved reference", k, "in", name)
            return k
        return labels[k]
    text = re.sub(r"\\ref\{([^}]+)\}", ref, text)
    text = re.sub(r"\\label\{[^}]+\}", "", text)
    text = re.sub(r"\\cmidrule(\([a-z]*\))?\{[^}]*\}", "", text)
    text = re.sub(r"\\todo\{", r"\\textbf{[TO DO: ", text)
    # appendices: unnumbered headings "Appendix A: ..."
    text = re.sub(r"\\section\{([^}]*)\}\s*(?=\n)", lambda m: m.group(0), text)
    return text


def build_sources():
    labels = read_labels()
    if TMP.exists():
        shutil.rmtree(TMP)
    TMP.mkdir()
    (TMP / "sections").mkdir()
    shutil.copytree(HERE / "generated", TMP / "generated")
    shutil.copy(HERE / "refs.bib", TMP / "refs.bib")
    shutil.copy(HERE / "ieee.csl", TMP / "ieee.csl")
    (TMP / "figures").mkdir()
    for f in (HERE / "figures").iterdir():
        if f.suffix == ".pdf":
            pix = pymupdf.open(f)[0].get_pixmap(dpi=170)
            pix.save(TMP / "figures" / (f.stem + ".png"))
        elif f.suffix == ".png":
            shutil.copy(f, TMP / "figures" / f.name)
    for f in (HERE / "sections").glob("*.tex"):
        text = transform(f.read_text(encoding="utf-8"), labels, f.name)
        if f.name.startswith("appendix_"):
            letter = "A" if f.name == "appendix_ai.tex" else "B"
            text = re.sub(r"\\section\{([^}]*)\}", lambda m: "\\section*{Appendix " + letter + ": " + m.group(1) + "}", text, count=1)
            text = text.replace("\\subsection{", "\\subsection*{")          # keep appendix sub-headings unnumbered
        (TMP / "sections" / f.name).write_text(text, encoding="utf-8")
    for f in (TMP / "generated").glob("tab_*.tex"):
        f.write_text(re.sub(r"\\cmidrule(\([a-z]*\))?\{[^}]*\}", "", f.read_text(encoding="utf-8")), encoding="utf-8")
    main = (HERE / "main.tex").read_text(encoding="utf-8")
    title = main[main.index("\\title{") + 6:]
    title = title[1:balanced(title, 0)]
    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", main, re.S).group(1)
    keywords = re.search(r"\\begin\{IEEEkeywords\}(.*?)\\end\{IEEEkeywords\}", main, re.S).group(1).strip()
    tex = r"""\documentclass{article}
\usepackage{graphicx}
\input{generated/numbers.tex}
\input{generated/numbers_analysis.tex}
\newcommand{\code}[1]{\texttt{#1}}
\newcommand{\repo}{\url{https://github.com/Zaid-fareed/23i-0829_C_GenAi_A1}}
\title{%s}
\author{Zaid Fareed \\ Roll No.\ 23I-0829, Section C --- Generative AI, Assignment 1 \\ Source code: \repo \\ Demonstration video: \url{https://youtu.be/xUows4eME8M}}
\date{}
\begin{document}
\maketitle
\begin{abstract}
%s
\end{abstract}
\noindent\textbf{Index Terms:} %s

""" % (title, re.sub(r"\\todo\{", r"\\textbf{[TO DO: ", abstract), keywords)
    for sec in ("intro", "related", "data", "task1", "task2", "task3", "task4", "optuna", "app", "limits", "appendix_ai"):
        tex += f"\\input{{sections/{sec}}}\n\n"
    tex += "\\end{document}\n"
    (TMP / "main_word.tex").write_text(tex, encoding="utf-8")


def make_reference_doc():
    ref = TMP / "ref.docx"
    ref.write_bytes(subprocess.run([pypandoc.get_pandoc_path(), "--print-default-data-file", "reference.docx"], capture_output=True, check=True).stdout)
    d = Document(ref)
    for s in d.styles:
        if s.font is not None and s.type == 1:
            s.font.name = "Times New Roman"
            rpr = s.element.get_or_add_rPr()
            rfonts = rpr.find(qn("w:rFonts"))
            if rfonts is None:
                rfonts = OxmlElement("w:rFonts"); rpr.append(rfonts)
            for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
                rfonts.set(qn(a), "Times New Roman")
    sizes = {"Normal": 10.5, "Body Text": 10.5, "First Paragraph": 10.5, "Compact": 9, "Title": 18, "Author": 10.5, "Abstract": 9.5,
             "Heading 1": 13, "Heading 2": 11.5, "Heading 3": 10.5, "Image Caption": 9, "Table Caption": 9, "Captioned Figure": 9, "Bibliography": 9}
    for name, pt in sizes.items():
        try:
            d.styles[name].font.size = Pt(pt)
        except KeyError:
            pass
    for name in ("Heading 1", "Heading 2", "Heading 3", "Title", "Author", "Abstract"):
        try:
            from docx.shared import RGBColor
            d.styles[name].font.color.rgb = RGBColor(0, 0, 0)
            d.styles[name].font.bold = name != "Abstract"
        except KeyError:
            pass
    for name in ("Body Text", "First Paragraph"):
        try:
            d.styles[name].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        except KeyError:
            pass
    sec = d.sections[0]
    sec.left_margin = sec.right_margin = Cm(2.0)
    sec.top_margin = sec.bottom_margin = Cm(2.0)
    d.save(ref)
    return ref


def postprocess():
    d = Document(OUT)
    sec = d.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)      # A4
    sec.left_margin = sec.right_margin = Cm(2.0)
    sec.top_margin = sec.bottom_margin = Cm(2.0)
    usable = sec.page_width - sec.left_margin - sec.right_margin
    for t in d.tables:                                  # booktabs-like borders, small font
        tblPr = t._tbl.tblPr
        borders = OxmlElement("w:tblBorders")
        for edge in ("top", "bottom", "insideH"):
            e = OxmlElement(f"w:{edge}")
            e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "4"); e.set(qn("w:color"), "808080")
            borders.append(e)
        tblPr.append(borders)
        for row in t.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    for r in p.runs:
                        r.font.size = Pt(8)
    for shape in d.inline_shapes:                       # never wider than the text area
        if shape.width > usable:
            ratio = usable / shape.width
            shape.width, shape.height = int(shape.width * ratio), int(shape.height * ratio)
    for p in d.paragraphs:
        if p.runs and any(r._element.xpath(".//w:drawing") for r in p.runs):
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    d.save(OUT)


def verify():
    d = Document(OUT)
    text = "\n".join(p.text for p in d.paragraphs)
    problems = [pat for pat in (r"\\n[A-Z][A-Za-z]+", r"fig:", r"tab:", r"sec:", r"\?\?", r"FAIL[A-Z]+") if re.search(pat, text)]
    print(f"docx: {len(d.paragraphs)} paragraphs, {len(d.tables)} tables, {len(d.inline_shapes)} pictures, {len(text.split())} words")
    print("leftover LaTeX/placeholder patterns:", problems or "none")
    todos = re.findall(r"\[TO DO:[^\]]*\]", text)
    print("TO DO markers for the author:", len(todos))


if __name__ == "__main__":
    build_sources()
    ref = make_reference_doc()
    pypandoc.convert_file(
        str(TMP / "main_word.tex"), "docx", outputfile=str(OUT), cworkdir=str(TMP),
        extra_args=["--citeproc", "--bibliography=refs.bib", "--csl=ieee.csl", "--number-sections", "--resource-path=.",
                    f"--reference-doc={ref}", "--metadata=link-citations:true", "--metadata=reference-section-title:References"])
    postprocess()
    verify()
    print("wrote", OUT)
