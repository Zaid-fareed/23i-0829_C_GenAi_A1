"""Split the tall Task 4 example figure (12 rows) into three strips of four rows so it fits one wide figure in the report.

  python report/split_task4_examples.py
Reads figures/task4_examples.png and writes figures/task4_examples_{a,b,c}.png. The overlapping title of the original
figure is replaced by a clean row of column labels that is repeated on every strip.
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FIG = Path(__file__).resolve().parent / "figures"
im = Image.open(FIG / "task4_examples.png").convert("RGB")
w, h = im.size
head = 36                                    # height of the title / column-header band in the original
row = (h - head) / 12
try:
    font = ImageFont.truetype("arial.ttf", 17)
except OSError:
    font = ImageFont.load_default()
labels = [("photo", 130), ("ground truth", 293), ("generated", 488), ("abs error", 669)]   # x centres (pixels)
for k, name in enumerate("abc"):
    top, bottom = round(head + 4 * k * row), round(head + 4 * (k + 1) * row)
    body = im.crop((0, top, w, bottom))
    out = Image.new("RGB", (w, 28 + body.height), "white")
    d = ImageDraw.Draw(out)
    for text, cx in labels:
        d.text((cx, 14), text, fill="black", font=font, anchor="mm")
    out.paste(body, (0, 28))
    out.save(FIG / f"task4_examples_{name}.png", optimize=True)
print("wrote task4_examples_a/b/c.png")
