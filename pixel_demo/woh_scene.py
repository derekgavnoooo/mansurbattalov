"""Кадр в манере World of Horror: 1 бит, 320×180 (×4), полутона — дизеринг Байера.

Всё рисуется без сглаживания, по пикселям: силуэты, контуры, узоры.
"""
import math
import sys
import numpy as np
from PIL import Image, ImageDraw

W, H = 320, 180
rng = np.random.default_rng(3)
yy, xx = np.mgrid[0:H, 0:W].astype(float)

B8 = np.array([[0, 32, 8, 40, 2, 34, 10, 42], [48, 16, 56, 24, 50, 18, 58, 26], [12, 44, 4, 36, 14, 46, 6, 38],
               [60, 28, 52, 20, 62, 30, 54, 22], [3, 35, 11, 43, 1, 33, 9, 41], [51, 19, 59, 27, 49, 17, 57, 25],
               [15, 47, 7, 39, 13, 45, 5, 37], [63, 31, 55, 23, 61, 29, 53, 21]]) / 64.0 + 1 / 128
TH = np.tile(B8, (H // 8 + 1, W // 8 + 1))[:H, :W]


def lowfreq(cell):
    g = Image.fromarray((rng.random((H // cell + 2, W // cell + 2)) * 255).astype(np.uint8))
    return np.asarray(g.resize((W + cell * 2, H + cell * 2), Image.BICUBIC), float)[:H, :W] / 255


NOISE = lowfreq(8)


class Pic:
    def __init__(self):
        self.a = np.zeros((H, W), np.uint8)

    @property
    def d(self):
        return ImageDraw.Draw(self.img)

    def tone(self, poly, t):
        """Залить многоугольник полутоном t (0 — чёрный, 1 — белый) или массивом тонов."""
        m = Image.new("1", (W, H), 0)
        ImageDraw.Draw(m).polygon([tuple(p) for p in poly], fill=1)
        m = np.asarray(m, bool)
        tt = t if np.ndim(t) else np.full((H, W), float(t))
        self.a[m] = ((tt > TH) * 255).astype(np.uint8)[m]
        return m

    def draw(self):
        im = Image.fromarray(self.a)
        return im, ImageDraw.Draw(im)

    def commit(self, im):
        self.a = np.asarray(im).copy()


def sky(p):
    cx, cy = 196, 46
    dx, dy = xx - cx, (yy - cy) / 0.36
    r = np.hypot(dx, dy)
    th = np.arctan2(dy, dx)
    sp = r / 24 + th / (2 * np.pi) * 2 + NOISE * 0.12
    band = np.sin(sp * 2 * np.pi)
    t = 0.40 + 0.22 * band + 0.16 * (NOISE - 0.5)
    t = np.where(band > 0.8, 0.03, t)
    t = np.where(r < 32 + 8 * NOISE, 1.0, t)
    t = np.where((r >= 32 + 8 * NOISE) & (r < 46), np.maximum(t, 0.78), t)
    t = t * (1 - 0.75 * np.clip((xx - 222) / 30, 0, 1))      # справа небо темнеет — за головой Рэй
    p.tone([(0, 0), (W, 0), (W, 101), (0, 101)], t)
    # море: тёмное, с белыми штрихами бликов, гуще у столпа света
    p.tone([(0, 101), (W, 101), (W, H), (0, H)], 0.06 + 0.06 * NOISE)
    im, d = p.draw()
    for _ in range(420):
        y = int(rng.uniform(102, 180))
        x = rng.normal(196, 60) if rng.random() < 0.6 else rng.uniform(0, 320)
        L = int(rng.uniform(1, 3 + (y - 100) / 7))
        d.line([(x, y), (x + L, y)], fill=255)
    d.line([(0, 101), (320, 101)], fill=255)
    p.commit(im)


def lighthouse(p, x=196, base=101, top=50):
    im, d = p.draw()
    p.commit(im)
    im, d = p.draw()
    h = base - top
    for y in range(top + 9, base):                          # конусная башня с белой гранью
        k = (y - top - 9) / (h - 9)
        w = 2.5 + 3.5 * k
        d.line([(x - w, y), (x + w, y)], fill=0)
        d.point((int(x - w), y), fill=255)
    for y in range(top + 16, base - 4, 9):                  # маленькие окна
        d.point([(x, y), (x, y + 1)], fill=255)
    d.line([(x - 5, top + 9), (x + 5, top + 9)], fill=0)    # галерея
    d.line([(x - 5, top + 8), (x + 5, top + 8)], fill=0)
    for k in range(-5, 6, 2):
        d.point((x + k, top + 7), fill=0)
    d.line([(x - 5, top + 6), (x + 5, top + 6)], fill=0)
    d.rectangle([x - 3, top + 2, x + 3, top + 5], fill=255, outline=0)   # фонарь
    d.line([(x, top + 2), (x, top + 5)], fill=0)
    d.line([(x - 3, top + 1), (x + 3, top + 1)], fill=0)
    d.line([(x - 2, top), (x + 2, top)], fill=0)
    d.point((x, top - 1), fill=0)
    d.polygon([(x - 9, base), (x - 5, base - 3), (x + 6, base - 3), (x + 11, base)], fill=0)   # скала
    d.point([(x - 6, base - 2), (x + 7, base - 2)], fill=255)
    p.commit(im)


def hills(p):
    hill = [(0, 66), (18, 62), (40, 68), (70, 76), (110, 86), (150, 95), (170, 101), (0, 101)]
    p.tone(hill, 0.0)
    im, d = p.draw()
    for _ in range(160):
        x = int(rng.uniform(0, 165))
        ytop = np.interp(x, [q[0] for q in hill[:-1]], [q[1] for q in hill[:-1]])
        d.point((x, int(ytop + rng.uniform(1, 14))), fill=255)
    d.line(hill[:-1], fill=255)
    p.commit(im)


def house3(p, x, y, w, ws, h, rh):
    """Дом в ракурсе 3/4: белый фасад, боковая стена в полутоне, черепичная крыша."""
    side = [(x + w, y), (x + w + ws, y - 5), (x + w + ws, y + h - 5), (x + w, y + h)]
    p.tone(side, 0.42)
    roof = [(x - 4, y + 1), (x + w + 2, y + 1), (x + w + ws + 4, y - 5), (x + w + ws - 2, y - 5 - rh),
            (x + 6, y - rh), ]
    p.tone(roof, 0.0)
    im, d = p.draw()
    d.rectangle([x, y, x + w, y + h], fill=255, outline=0)
    d.polygon(side, outline=0)
    for k in range(2, rh + 3, 3):                           # ряды черепицы
        f = k / (rh + 4)
        xa = x - 4 + (10 * f)
        xb = x + w + ws + 4 - 6 * f
        ya, yb = y + 1 - (rh + 1) * f, y - 5 - (rh) * f
        n = int(xb - xa)
        for i in range(0, n, 2):
            d.point((int(xa + i + (k // 3) % 2), int(ya + (yb - ya) * i / max(n, 1))), fill=255)
    d.line([(x - 4, y + 1), (x + w + 2, y + 1), (x + w + ws + 4, y - 5)], fill=255)   # белый карниз
    d.line([(x + w + 2, y + 1), (x + w + ws - 2, y - 5 - rh)], fill=255)
    for wx in range(x + 4, x + w - 8, 14):                  # окна с переплётом
        d.rectangle([wx, y + 6, wx + 8, y + 13], fill=0)
        d.line([(wx + 4, y + 6), (wx + 4, y + 13)], fill=255)
        d.line([(wx, y + 9), (wx + 8, y + 9)], fill=255)
        d.line([(wx - 1, y + 14), (wx + 9, y + 14)], fill=0)
    d.rectangle([x + w - 7, y + h - 12, x + w - 2, y + h], fill=0)   # дверь
    d.point((x + w - 3, y + h - 6), fill=255)
    p.commit(im)


def town(p):
    ground = [(0, 116), (60, 118), (120, 124), (180, 132), (222, 148), (222, 180), (0, 180)]
    p.tone(ground, 0.0)
    im, d = p.draw()
    d.line(ground[:5], fill=255)
    d.rectangle([2, 102, 24, 120], fill=0, outline=255)     # дальний дом-многоэтажка
    for wy in range(105, 119, 4):
        for wx in range(5, 23, 4):
            if rng.random() < 0.8:
                d.rectangle([wx, wy, wx + 1, wy + 1], fill=255)
    p.commit(im)
    house3(p, 6, 142, 40, 12, 38, 11)
    house3(p, 66, 130, 30, 10, 50, 10)
    house3(p, 114, 150, 50, 14, 30, 12)
    im, d = p.draw()
    for x0 in range(0, 214, 3):                             # забор
        y0 = 176
        d.line([(x0, y0), (x0, y0 + 4)], fill=255)
    d.line([(0, 176), (214, 176)], fill=255)
    for cx_, cy_, r_ in ((58, 160, 7), (104, 168, 9), (170, 170, 8)):   # кусты: чёрные кляксы с бликами
        d.ellipse([cx_ - r_, cy_ - r_, cx_ + r_, cy_ + r_], fill=0)
        for _ in range(12):
            a = rng.uniform(math.pi, 2 * math.pi)
            d.point((cx_ + r_ * 0.8 * math.cos(a) + rng.normal(0, 1), cy_ + r_ * 0.8 * math.sin(a) + rng.normal(0, 1)), fill=255)
    # столб с проводами
    d.line([(204, 60), (204, 180)], fill=0)
    d.line([(205, 60), (205, 180)], fill=0)
    d.line([(198, 66), (211, 66)], fill=0)
    d.line([(199, 72), (210, 72)], fill=0)
    for yw, sag in ((66, 10), (72, 13)):
        pts = [(204 - t, yw + sag * (1 - ((t - 102) / 102) ** 2) * 0.9) for t in range(0, 205, 4)]
        d.line(pts, fill=0)
    p.commit(im)


def rei(p):
    hair = [(225, 124), (222, 98), (226, 74), (236, 58), (250, 48), (266, 44), (282, 47), (296, 56), (305, 72),
            (309, 94), (309, 114), (306, 130), (300, 120), (296, 133), (292, 118), (240, 118), (236, 133), (231, 122),
            (226, 131)]
    face = [(238, 84), (294, 84), (295, 100), (291, 114), (282, 126), (270, 134), (262, 134), (250, 126), (241, 114),
            (237, 100)]
    neck = [(256, 126), (276, 126), (277, 148), (255, 148)]
    body = [(214, 180), (222, 158), (240, 150), (256, 146), (276, 146), (294, 150), (312, 160), (320, 170), (320, 180)]
    tips = [(237, 84), (241, 97), (246, 88), (251, 96), (257, 87), (263, 103), (268, 88), (274, 96), (280, 87),
            (286, 96), (291, 88), (295, 84)]
    bangs = tips + [(298, 62), (282, 52), (266, 50), (250, 53), (236, 62)]
    hair_t = np.clip(0.86 - np.clip((xx - 262) / 46, 0, 1) * 0.5 + 0.05 * (NOISE - 0.5), 0, 1)
    # блик на волосах — светлая дуга
    arc = np.abs(np.hypot(xx - 266, (yy - 100) * 1.2) - 46) < 2.2
    hair_t = np.where(arc & (yy < 70), 0.97, hair_t)
    p.tone(hair, hair_t)
    p.tone(body, 0.0)
    p.tone(neck, 0.45)
    face_t = np.clip(1.0 - np.clip((xx - 284) / 12, 0, 1) * 0.35, 0, 1)
    p.tone(face, face_t)
    p.tone([(t[0], t[1] + 3) for t in tips] + [(295, 84), (237, 84)], 0.72)   # тень под чёлкой
    p.tone(bangs, hair_t)
    im, d = p.draw()
    outer = hair[:15]
    d.line([(x_ + 1, y_) if x_ > 266 else (x_ - 1, y_) for x_, y_ in outer[2:13]], fill=255)   # белый ореол
    d.line(outer, fill=0)
    d.line([(x_ + (1 if x_ > 266 else 0), y_) for x_, y_ in outer[6:13]], fill=0)
    d.line(hair[16:] + [hair[0]], fill=0)
    d.line(hair[:6], fill=255)                               # контровой свет от «ока»
    d.line(tips, fill=0)
    for (x_, y_) in tips[1::2]:                              # пряди
        d.line([(x_ - 3, 56), (x_ - 1, 74), (x_, y_ - 3)], fill=0)
    for x_ in (229, 234, 300, 305):
        d.line([(x_, 92), (x_ + (1 if x_ < 260 else -1), 120)], fill=0)
    d.line([(237, 86), (237, 100), (241, 114), (244, 118)], fill=0)
    d.line([(295, 86), (295, 100), (291, 114), (288, 118)], fill=0)
    d.line([(241, 114), (250, 126), (262, 134), (270, 134), (282, 126), (291, 114)], fill=0)
    # шея, тело
    d.line([(255, 126), (255, 148)], fill=0)
    d.line([(277, 126), (277, 148)], fill=0)
    d.line(body[:3], fill=255)
    d.polygon([(256, 146), (266, 160), (276, 146), (284, 150), (266, 168), (248, 150)], fill=255, outline=0)
    d.line([(248, 150), (238, 180)], fill=255)
    d.line([(284, 150), (294, 180)], fill=255)
    d.polygon([(264, 160), (268, 160), (270, 178), (266, 180), (262, 178)], fill=0)
    d.line([(264, 161), (264, 177)], fill=255)
    # глаза: толстое верхнее веко, крупный зрачок с бликом, нижнее веко пунктиром
    for ex, far in ((252, False), (281, True)):
        d.line([(ex - 7, 104), (ex - 5, 101), (ex - 2, 100), (ex + 3, 100), (ex + 6, 102)], fill=0)
        d.line([(ex - 6, 104), (ex - 4, 102), (ex + 4, 101), (ex + 6, 103)], fill=0)
        d.line([(ex + 6, 102), (ex + 8, 100)], fill=0)
        d.ellipse([ex - 3, 101, ex + 2, 110], fill=0)
        d.point([(ex - 2, 103), (ex - 1, 103), (ex - 2, 104)], fill=255)
        d.point([(ex + 1, 108)], fill=255)
        for k in range(-4, 5, 2):
            d.point((ex + k, 111 + (abs(k) > 3)), fill=0)
    # брови «домиком» — тревога
    d.line([(246, 95), (252, 93)], fill=0)
    d.line([(280, 93), (286, 95)], fill=0)
    # румянец штрихами, нос, рот
    for bx in (245, 248, 284, 287):
        d.point([(bx, 118), (bx + 1, 117)], fill=0)
    d.point([(267, 117), (266, 118)], fill=0)
    d.line([(263, 125), (269, 125)], fill=0)
    d.point([(262, 126), (270, 126)], fill=0)
    d.point((266, 126), fill=0)
    # заколка-крестик
    d.line([(244, 66), (249, 71)], fill=0)
    d.line([(249, 66), (244, 71)], fill=0)
    d.line([(244, 65), (249, 70)], fill=255)
    p.commit(im)


def render():
    p = Pic()
    sky(p)
    lighthouse(p)
    hills(p)
    town(p)
    rei(p)
    return Image.fromarray(p.a)


if __name__ == "__main__":
    render().resize((W * 4, H * 4), Image.NEAREST).save(sys.argv[1])
