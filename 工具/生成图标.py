"""从桌面版 icon.png 生成 Android 图标（传统 mipmap + adaptive icon）。

桌面版图标是「深绿底 + 白色图案 + 圆角」，四角透明。
- 传统 mipmap：整张图直接缩放到各密度尺寸。
- adaptive icon：背景用纯绿 #036954，前景只保留白色图案（按亮度转 alpha），
  内容缩到 108dp 画布的约 62%，落在安全区内。
"""
import os
from PIL import Image

SRC = r"D:\Audivox事务\Hearizon Sync\icon.png"
RES = r"D:\Audivox事务\Hearizon Sync手机版\app\src\main\res"
BG = (3, 105, 84)

src = Image.open(SRC).convert("RGBA")
W, H = src.size

# ---------- 1. 传统 mipmap ----------
sizes = {"mdpi": 48, "hdpi": 72, "xhdpi": 96, "xxhdpi": 144, "xxxhdpi": 192}
for dpi, size in sizes.items():
    d = os.path.join(RES, f"mipmap-{dpi}")
    os.makedirs(d, exist_ok=True)
    src.resize((size, size), Image.LANCZOS).save(os.path.join(d, "ic_launcher.png"))
    src.resize((size, size), Image.LANCZOS).save(os.path.join(d, "ic_launcher_round.png"))
print("传统 mipmap 完成:", ", ".join(sizes))

# ---------- 2. adaptive icon 前景（只留白色图案）----------
fg = Image.new("RGBA", (W, H), (0, 0, 0, 0))
sp = src.load()
fp = fg.load()
for y in range(H):
    for x in range(W):
        r, g, b, a = sp[x, y]
        if a == 0:
            continue
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        # 绿色底 lum≈71，白色 lum=255；90 以下全透明，255 全不透明
        k = (lum - 90.0) / (255.0 - 90.0)
        if k <= 0:
            continue
        fp[x, y] = (255, 255, 255, int(max(0.0, min(1.0, k)) * a))

# 裁到内容外接框，再按 62% 缩放进 108dp 画布
bbox = fg.getbbox()
content = fg.crop(bbox)
side = max(content.size)
square = Image.new("RGBA", (side, side), (0, 0, 0, 0))
square.paste(content, ((side - content.size[0]) // 2, (side - content.size[1]) // 2))

for dpi, canvas in {"mdpi": 108, "hdpi": 162, "xhdpi": 216, "xxhdpi": 324, "xxxhdpi": 432}.items():
    d = os.path.join(RES, f"drawable-{dpi}")
    os.makedirs(d, exist_ok=True)
    inner = int(canvas * 0.62)
    art = square.resize((inner, inner), Image.LANCZOS)
    out = Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0))
    off = (canvas - inner) // 2
    out.paste(art, (off, off), art)
    out.save(os.path.join(d, "ic_launcher_foreground.png"))
print("adaptive 前景完成")

# ---------- 3. adaptive icon 描述 ----------
anydpi = os.path.join(RES, "mipmap-anydpi-v26")
os.makedirs(anydpi, exist_ok=True)
xml = """<?xml version="1.0" encoding="utf-8"?>
<adaptive-icon xmlns:android="http://schemas.android.com/apk/res/android">
    <background android:drawable="@color/ic_launcher_background" />
    <foreground android:drawable="@drawable/ic_launcher_foreground" />
</adaptive-icon>
"""
for name in ("ic_launcher.xml", "ic_launcher_round.xml"):
    with open(os.path.join(anydpi, name), "w", encoding="utf-8") as f:
        f.write(xml)

vals = os.path.join(RES, "values")
os.makedirs(vals, exist_ok=True)
with open(os.path.join(vals, "ic_launcher_background.xml"), "w", encoding="utf-8") as f:
    f.write('<?xml version="1.0" encoding="utf-8"?>\n<resources>\n'
            f'    <color name="ic_launcher_background">#036954</color>\n</resources>\n')
print("adaptive icon XML 完成")
