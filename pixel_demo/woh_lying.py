"""1-бит, вид сверху: Рэй лежит на мокром асфальте, в луже отражается вихрь с маяком.

Персонаж рисуется в своих координатах (лицо «прямо»), затем весь рисунок поворачивается —
так получается заваленный горизонт и ракурс сверху.
"""
import math
import sys
import numpy as np
from PIL import Image, ImageDraw
from woh_scene import Pic, W, H, xx, yy, TH, lowfreq

rng = np.random.default_rng(11)
N1, N2 = lowfreq(6), lowfreq(16)

CX, CY, ANG, S = 138, 96, math.radians(-22), 1.2


def R(pts):
    """Локальные координаты головы → кадр (поворот + масштаб)."""
    c, s = math.cos(ANG), math.sin(ANG)
    return [(CX + S * (u * c - v * s), CY + S * (u * s + v * c)) for u, v in pts]


def loc():
    """Обратное преобразование для массивов кадра → (u, v)."""
    c, s = math.cos(ANG), math.sin(ANG)
    dx, dy = (xx - CX) / S, (yy - CY) / S
    return dx * c + dy * s, -dx * s + dy * c


def ground(p):
    t = 0.07 + 0.10 * N1 + 0.05 * (N2 - 0.5)                  # мокрый асфальт
    p.tone([(0, 0), (W, 0), (W, H), (0, H)], np.clip(t, 0, 1))
    im, d = p.draw()
    for _ in range(500):                                      # крошка асфальта
        d.point((rng.integers(0, W), rng.integers(0, H)), fill=255 if rng.random() < 0.6 else 0)
    p.commit(im)
    # лужа с отражением вихря и перевёрнутого маяка
    pcx, pcy = 258, 46
    pud = [(pcx + (52 + 8 * math.sin(3 * a) + 5 * math.cos(5 * a)) * math.cos(a),
            pcy + (30 + 5 * math.sin(4 * a)) * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 40, endpoint=False)]
    dx, dy = xx - (pcx + 6), (yy - (pcy + 4)) / 0.5
    r = np.hypot(dx, dy)
    th = np.arctan2(dy, dx)
    band = np.sin((r / 9 - th / (2 * np.pi) * 2 + N1 * 0.1) * 2 * np.pi)
    t = 0.35 + 0.2 * band
    t = np.where(band > 0.8, 0.05, t)
    t = np.where(r < 12 + 3 * N1, 1.0, t)
    p.tone(pud, t)
    im, d = p.draw()
    d.polygon(pud, outline=255)
    lx, ly = pcx + 6, pcy + 4                                 # маяк отражён вверх ногами
    for y in range(ly, ly + 22):
        w = 1.2 + 1.6 * (1 - (y - ly) / 22)
        d.line([(lx - w, y), (lx + w, y)], fill=0)
    d.rectangle([lx - 2, ly + 22, lx + 2, ly + 25], fill=255, outline=0)
    for _ in range(6):                                        # круги от капель
        ex, ey = pcx + rng.normal(0, 22), pcy + rng.normal(0, 10)
        rr = rng.uniform(2, 5)
        d.ellipse([ex - rr, ey - rr * 0.5, ex + rr, ey + rr * 0.5], outline=255)
    p.commit(im)


def hair_spread():
    """Каре, рассыпавшееся по асфальту: волнистый ореол + несколько выбившихся прядок."""
    pts = []
    for a in np.linspace(math.radians(150), math.radians(390), 90):
        r = 60 + 5 * math.sin(a * 5 + 0.7) + 3 * math.sin(a * 11 + 2.0) + 2.5 * math.sin(a * 17)
        for ac, L, wdt in ((200, 14, 0.08), (262, 11, 0.07), (318, 16, 0.08), (352, 9, 0.06)):   # выбившиеся прядки
            d_ = (a - math.radians(ac) + math.pi) % (2 * math.pi) - math.pi
            r += L * max(0.0, 1 - abs(d_) / wdt) ** 1.6
        pts.append((r * math.cos(a), r * math.sin(a)))
    return pts


def rei(p):
    u, v = loc()
    face = [(-33, -20), (33, -20), (34, 0), (30, 18), (20, 34), (8, 44), (-8, 44), (-20, 34), (-30, 18), (-34, 0)]
    tips = [(-35, -20), (-31, -8), (-26, -17), (-21, -9), (-14, -18), (-7, -9), (-2, -17), (2, -4), (7, -17), (13, -9),
            (20, -18), (26, -9), (31, -17), (35, -20)]
    top = [(37, -40), (22, -53), (0, -57), (-22, -53), (-37, -40)]
    spread = hair_spread()
    body = [(-64, 58), (-30, 50), (30, 50), (64, 58), (90, 150), (-90, 150)]
    neck = [(-10, 38), (10, 38), (12, 58), (-12, 58)]
    sleeve = [(40, 56), (60, 50), (76, 18), (80, -8), (68, -12), (62, 12), (46, 40)]
    hand = [(68, -12), (80, -8), (86, -16), (92, -24), (88, -28), (82, -24), (84, -34), (79, -36), (76, -28),
            (74, -36), (69, -35), (70, -26), (65, -30), (62, -26), (66, -18)]
    hair_t = np.clip(0.8 - 0.25 * np.clip((u + 10) / 60, 0, 1) + 0.08 * (N1 - 0.5), 0, 1)
    p.tone(R(spread), hair_t)
    p.tone(R(body), 0.0)
    p.tone(R(sleeve), 0.0)
    p.tone(R(neck), 0.42)
    face_t = np.clip(1.0 - 0.32 * np.clip((u - 18) / 14, 0, 1), 0, 1)
    p.tone(R(face), face_t)
    p.tone(R([(x_, y_ + 4) for x_, y_ in tips] + [(35, -20), (-35, -20)]), 0.7)   # тень под чёлкой
    p.tone(R(tips + top), hair_t)
    p.tone(R(hand), 1.0)
    im, d = p.draw()
    L = lambda pts, f=0: d.line(R(pts), fill=f)
    # пряди на асфальте
    sp = R(spread)
    d.line(sp, fill=0)
    for i in range(3, len(spread) - 3, 6):                    # изогнутые линии прядей
        a = math.atan2(spread[i][1], spread[i][0])
        rr = math.hypot(*spread[i])
        curl = 0.12 * math.sin(i * 1.7)
        seg = [((46 + k * (rr - 50) / 4) * math.cos(a + curl * k / 4), (46 + k * (rr - 50) / 4) * math.sin(a + curl * k / 4)) for k in range(5)]
        L(seg, 0 if i % 12 else 255)
    # тело: белые кромки, воротник, галстук
    L(body[:5], 255)
    L([(-30, 50), (0, 66), (30, 50)])
    d.polygon(R([(-12, 52), (0, 66), (12, 52), (20, 56), (0, 76), (-20, 56)]), fill=255, outline=0)
    d.polygon(R([(-3, 66), (3, 66), (4, 96), (0, 100), (-4, 96)]), fill=0)
    L([(-20, 56), (-38, 150)], 255)
    L([(20, 56), (38, 150)], 255)
    # рукав и кисть (ладонь вверх, пальцы расслаблены)
    L(sleeve, 255)
    d.polygon(R(hand), outline=0)
    L([(72, -22), (77, -26)])
    L([(68, -12), (80, -8)], 255)
    # лицо
    L([(-34, -18), (-34, 0), (-30, 18), (-20, 34), (-8, 44), (8, 44), (20, 34), (30, 18), (34, 0), (34, -18)])
    L([(-12, 38), (-12, 56)])
    L([(12, 38), (12, 56)])
    L(tips)
    for (x_, y_) in tips[1::2]:
        L([(x_ - 3, -52), (x_ - 1, -34), (x_, y_ - 4)])
    # глаза: широко открыты, маленькие зрачки — пустой взгляд вверх
    for ex in (-14, 14):
        L([(ex - 9, 0), (ex - 6, -4), (ex, -5), (ex + 6, -4), (ex + 9, -1)])
        L([(ex - 8, -1), (ex - 5, -4), (ex + 5, -4), (ex + 8, -1)])
        L([(ex - 7, 6), (ex - 3, 8), (ex + 3, 8), (ex + 7, 5)])
        cx_, cy_ = R([(ex, 2)])[0]
        d.ellipse([cx_ - 2.2, cy_ - 2.2, cx_ + 2.2, cy_ + 2.2], fill=0)
        d.point((round(cx_ - 1), round(cy_ - 1)), fill=255)
        L([(ex + (6 if ex > 0 else -6), -3), (ex + (10 if ex > 0 else -10), -6)])
    L([(-21, -10), (-8, -12)])
    L([(8, -12), (21, -10)])
    for bx in (-22, -18, 18, 22):                             # румянец
        L([(bx, 14), (bx + 2, 12)])
    L([(1, 14), (0, 16)])
    mo = R([(0, 28)])[0]
    d.ellipse([mo[0] - 2.5, mo[1] - 1.5, mo[0] + 2.5, mo[1] + 1.5], fill=0)
    # капли на лице
    for (du, dv) in ((-24, 22), (10, 30), (26, 6)):
        q = R([(du, dv)])[0]
        d.point((round(q[0]), round(q[1])), fill=0)
        d.point((round(q[0]) - 1, round(q[1]) - 1), fill=0)
    # заколка-крестик
    L([(-26, -36), (-20, -30)])
    L([(-20, -36), (-26, -30)])
    p.commit(im)


def rain(p):
    im, d = p.draw()
    for _ in range(110):
        x, y = rng.uniform(0, W), rng.uniform(0, H)
        Lr = rng.uniform(2, 5)
        d.line([(x, y), (x + Lr * 0.35, y + Lr)], fill=255 if rng.random() < 0.8 else 0)
    for _ in range(14):
        x, y, rr = rng.uniform(0, W), rng.uniform(0, H), rng.uniform(2, 4)
        d.ellipse([x - rr, y - rr * 0.6, x + rr, y + rr * 0.6], outline=255)
    p.commit(im)


def render():
    p = Pic()
    ground(p)
    rei(p)
    rain(p)
    return Image.fromarray(p.a)


if __name__ == "__main__":
    render().resize((W * 4, H * 4), Image.NEAREST).save(sys.argv[1])
