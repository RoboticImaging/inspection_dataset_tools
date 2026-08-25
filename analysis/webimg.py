"""Downscaled copies of the figures for embedding in the HTML report."""
from pathlib import Path
from PIL import Image

SRC = Path(__file__).resolve().parent / "figures"
DST = Path(__file__).resolve().parent / "figures_web"
DST.mkdir(exist_ok=True)
MAXW = 1700

for p in sorted(SRC.glob("*.png")):
    im = Image.open(p).convert("RGB")
    if im.width > MAXW:
        im = im.resize((MAXW, round(im.height * MAXW / im.width)), Image.LANCZOS)
    out = DST / (p.stem + ".jpg")
    im.save(out, "JPEG", quality=86, optimize=True, progressive=True)
    print(f"{p.name:52s} {p.stat().st_size/1024:7.0f}K -> {out.stat().st_size/1024:6.0f}K")
total = sum(f.stat().st_size for f in DST.glob("*.jpg"))
print(f"\ntotal {total/1024/1024:.2f} MB  (base64 ~{total*1.34/1024/1024:.2f} MB)")
