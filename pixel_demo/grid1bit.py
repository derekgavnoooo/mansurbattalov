"""Рендер ручной 1-битной сетки. Символы: # чёрный, . белый, : шахматка 50%, - точки 25%, = фон 75%."""
import sys
from PIL import Image, ImageDraw


def load(path, w=64):
    rows = open(path).read().rstrip("\n").split("\n")
    bad = [(i, len(r)) for i, r in enumerate(rows) if len(r) != w]
    if bad:
        print("строки не той длины:", bad)
    return [r.ljust(w, "=")[:w] for r in rows]


def is_black(c, x, y):
    return (c == "#" or (c == ":" and (x + y) % 2 == 0) or (c == "-" and x % 2 == 0 and y % 2 == 0)
            or (c == "=" and not (x % 2 == 0 and y % 2 == 0)))


def render(rows, scale=10, grid=False):
    h, w = len(rows), len(rows[0])
    im = Image.new("L", (w, h), 255)
    px = im.load()
    for y, r in enumerate(rows):
        for x, c in enumerate(r):
            px[x, y] = 0 if is_black(c, x, y) else 255
    im = im.resize((w * scale, h * scale), Image.NEAREST).convert("RGB")
    if grid:
        d = ImageDraw.Draw(im)
        for k in range(0, w + 1, 8):
            d.line([(k * scale, 0), (k * scale, h * scale)], fill=(255, 0, 0))
        for k in range(0, h + 1, 8):
            d.line([(0, k * scale), (w * scale, k * scale)], fill=(255, 0, 0))
    return im


if __name__ == "__main__":
    rows = load(sys.argv[1])
    render(rows, grid=len(sys.argv) > 3).save(sys.argv[2])
