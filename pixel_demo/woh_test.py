"""Тест в манере World of Horror: 1-бит, 320×180, полутона — упорядоченный дизеринг (Байер 8×8)."""
import math
import sys
import numpy as np
from PIL import Image, ImageDraw

W, H = 320, 180
rng = np.random.default_rng(3)

B8 = np.array([[0, 32, 8, 40, 2, 34, 10, 42], [48, 16, 56, 24, 50, 18, 58, 26], [12, 44, 4, 36, 14, 46, 6, 38],
               [60, 28, 52, 20, 62, 30, 54, 22], [3, 35, 11, 43, 1, 33, 9, 41], [51, 19, 59, 27, 49, 17, 57, 25],
               [15, 47, 7, 39, 13, 45, 5, 37], [63, 31, 55, 23, 61, 29, 53, 21]]) / 64.0 + 1 / 128


def dither(tone):
    th = np.tile(B8, (H // 8 + 1, W // 8 + 1))[:H, :W]
    return (tone > th).astype(np.uint8) * 255


tone = np.zeros((H, W))
yy, xx = np.mgrid[0:H, 0:W].astype(float)

# --- небо: вихрь (сплющенный в перспективе) ---
cx, cy = 200, 50
dx, dy = (xx - cx), (yy - cy) / 0.38
r = np.hypot(dx, dy)
th = np.arctan2(dy, dx)
spiral = r / 26 + th / (2 * np.pi) * 2
band = np.sin(spiral * 2 * np.pi)
noise = np.asarray(Image.fromarray((rng.random((23, 40)) * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC)) / 255
sky = 0.42 + 0.22 * band + 0.18 * (noise - 0.5)
sky = np.where(band > 0.82, 0.04, sky)                     # чёрные «рукава» вихря
sky = np.where(r < 34 + 6 * noise, 1.0, sky)               # светлое «око»
sky = np.where((r >= 34) & (r < 44), np.minimum(sky, 0.7) + 0.2, sky)
tone[:] = sky

# --- море ---
sea = yy >= 101
tone[sea] = 0.07 + 0.07 * noise[sea]
d = Image.fromarray(np.zeros((H, W), np.uint8))

img = Image.fromarray(dither(np.clip(tone, 0, 1)))
dr = ImageDraw.Draw(img)
for _ in range(260):                                       # блики волн — короткие белые штрихи
    y = int(rng.uniform(102, 180))
    x = int(rng.uniform(0, 320))
    L = int(rng.uniform(2, 4 + (y - 100) / 6))
    dr.line([(x, y), (x + L, y)], fill=255)
dr.line([(0, 101), (320, 101)], fill=255)

# --- маяк в столпе света ---
for y in range(54, 101):
    w = 2 + (y - 54) / 22
    dr.line([(200 - w, y), (200 + w, y)], fill=0)
dr.rectangle([197, 50, 203, 54], fill=0)
dr.point([(199, 52), (200, 52), (201, 52)], fill=255)
dr.line([(196, 101), (205, 101)], fill=0)

# --- берег и холм слева ---
hill = [(0, 70), (20, 66), (45, 72), (80, 80), (120, 88), (160, 97), (175, 101), (0, 101)]
dr.polygon(hill, fill=0)
for _ in range(120):                                       # деревья — белые искры по контуру
    x = int(rng.uniform(0, 170))
    y = int(np.interp(x, [p[0] for p in hill[:-1]], [p[1] for p in hill[:-1]]) + rng.uniform(1, 12))
    dr.point((x, y), fill=255)
dr.line(hill[:-1], fill=255)

# --- земля и дома на переднем плане ---
ground = [(0, 118), (60, 120), (120, 128), (180, 136), (215, 150), (215, 180), (0, 180)]
gm = Image.new("1", (W, H), 0); ImageDraw.Draw(gm).polygon(ground, fill=1)
g = np.asarray(gm, bool)
cur = np.asarray(img).copy()
cur[g] = 0
img = Image.fromarray(cur)
dr = ImageDraw.Draw(img)
dr.line(ground[:5], fill=255)


def house(x, y, w, h, roof_h):
    roof = [(x - 5, y), (x + 4, y - roof_h), (x + w - 4, y - roof_h), (x + w + 5, y)]
    dr.polygon(roof, fill=0)
    for k in range(2, roof_h, 3):                         # ряды черепицы — белым пунктиром
        yk = y - roof_h + k
        x0_, x1_ = x + 4 - k * 9 / roof_h, x + w - 4 + k * 9 / roof_h
        for xx_ in range(int(x0_), int(x1_), 2):
            dr.point((xx_ + (k // 3) % 2, yk), fill=255)
    dr.line(roof, fill=255)
    dr.rectangle([x, y, x + w, y + h], fill=255, outline=0)
    dr.line([(x, y + 1), (x + w, y + 1)], fill=0)
    dr.rectangle([x + 4, y + 6, x + 11, y + 12], fill=0)
    dr.line([(x + 7, y + 6), (x + 7, y + 12)], fill=255)
    if w > 30:
        dr.rectangle([x + w - 15, y + 6, x + w - 6, y + 12], fill=0)
        dr.line([(x + w - 10, y + 6), (x + w - 10, y + 12)], fill=255)
    for k in range(x + w // 2 - 4, x + w // 2 + 4):        # тень под карнизом
        dr.point((k, y + 2), fill=0)


dr.rectangle([2, 104, 22, 122], fill=0, outline=255)        # дальний многоквартирный дом
for wy in range(107, 121, 4):
    for wx in range(5, 21, 4):
        dr.rectangle([wx, wy, wx + 1, wy + 1], fill=255)
house(10, 146, 44, 34, 12)
house(62, 136, 34, 44, 10)
house(108, 152, 54, 28, 13)

# --- Рэй крупным планом справа ---
hair = [(226, 112), (226, 86), (230, 71), (238, 61), (250, 53), (264, 50), (278, 53), (290, 61), (297, 73), (300, 88),
        (301, 104), (300, 120), (296, 131), (291, 124), (286, 133), (282, 118), (242, 118), (237, 131), (233, 122), (227, 129)]
face = [(236, 86), (289, 86), (289, 99), (285, 112), (276, 124), (264, 131), (258, 131), (248, 124), (239, 112), (235, 99)]
neck = [(253, 122), (271, 122), (272, 141), (252, 141)]
body = [(212, 180), (220, 152), (238, 143), (254, 139), (272, 139), (290, 143), (308, 153), (318, 180)]
tips = [(236, 86), (240, 99), (245, 89), (251, 101), (256, 90), (261, 104), (266, 90), (271, 100), (277, 89),
        (283, 98), (289, 87)]
bangs = tips + [(290, 66), (276, 57), (262, 55), (248, 58), (238, 66)]


def mask(poly):
    m = Image.new("1", (W, H), 0)
    ImageDraw.Draw(m).polygon(poly, fill=1)
    return np.asarray(m, bool)


mh, mf, mn, mb, mbg = mask(hair), mask(face), mask(neck), mask(body), mask(bangs)
t = np.zeros((H, W))
t[mh] = np.clip(0.97 - np.clip((xx[mh] - 268) / 34, 0, 1) * 0.55, 0, 1)
t[mn] = 0.4
t[mf] = np.clip(1.0 - np.clip((xx[mf] - 272) / 18, 0, 1) * 0.35, 0, 1)
t[mbg] = np.clip(0.97 - np.clip((xx[mbg] - 268) / 34, 0, 1) * 0.55, 0, 1)
t[mb] = 0.0
cur = np.asarray(img).copy()
lay = dither(t)
for m in (mh, mn, mf, mbg, mb):
    cur[m] = lay[m]
img = Image.fromarray(cur)
dr = ImageDraw.Draw(img)
dr.polygon(hair, outline=0)
dr.line(hair[:5], fill=255)                                 # контровой свет слева
dr.line(body[:-1], fill=255)
dr.line([(252, 122), (252, 141)], fill=0)
dr.line([(272, 122), (272, 141)], fill=0)
dr.line([(253, 136), (271, 136)], fill=0)                  # тень под подбородком
# линия лица
dr.line([(235, 99), (239, 112), (248, 124), (258, 131), (264, 131), (276, 124), (285, 112), (289, 99)], fill=0)
# чёлка
dr.line(tips, fill=0)
for (x, y) in tips[1::2]:
    dr.line([(x - 2, 62), (x, y - 4)], fill=0)
dr.line([(262, 56), (261, 98)], fill=0)
# брови (видны между прядями) — тревожные, домиком
dr.line([(244, 97), (249, 95)], fill=0)
dr.line([(275, 95), (280, 97)], fill=0)
dr.line([(236, 88), (235, 99), (238, 110), (241, 118)], fill=0)
dr.line([(289, 88), (289, 99), (287, 110), (283, 118)], fill=0)
# глаза
for ex in (250, 276):
    dr.line([(ex - 7, 104), (ex - 4, 101), (ex + 3, 101), (ex + 6, 103)], fill=0)
    dr.line([(ex - 6, 103), (ex - 4, 102), (ex + 4, 102), (ex + 5, 103)], fill=0)
    dr.rectangle([ex - 2, 103, ex + 2, 109], fill=0)
    dr.point([(ex - 1, 104), (ex, 104), (ex - 1, 105)], fill=255)
    dr.point([(ex - 4, 109), (ex - 2, 110), (ex, 110), (ex + 2, 110), (ex + 3, 109)], fill=0)
dr.line([(243, 104), (241, 102)], fill=0)
dr.line([(283, 104), (285, 102)], fill=0)
# румянец штрихами
for bx in (244, 247, 278, 281):
    dr.point([(bx, 115), (bx + 1, 114)], fill=0)
dr.point([(263, 115), (264, 116)], fill=0)                 # нос
dr.line([(259, 122), (265, 122)], fill=0)                  # рот
dr.point([(258, 123), (266, 123)], fill=0)
# воротник, галстук
dr.polygon([(254, 139), (262, 151), (270, 139), (277, 143), (263, 157), (247, 143)], fill=255, outline=0)
dr.polygon([(261, 151), (264, 151), (265, 172), (262, 175), (260, 172)], fill=255, outline=0)
# заколка-крестик
dr.line([(242, 72), (246, 76)], fill=0)
dr.line([(246, 72), (242, 76)], fill=0)

big = img.resize((W * 4, H * 4), Image.NEAREST)
big.save(sys.argv[1])
