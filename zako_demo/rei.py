"""Рэй в простой манере клипа «雑魚»: одна ровная линия, плоская заливка, длинные рукава.

Координаты головы: (0,0) — между глаз; ширина лица ~170. Каждая поза — отдельный рисунок.
"""
import math
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "daikirai_demo"))
from flat import catmull, path, ellipse, hexc, Xf, resample_n  # noqa: E402

LINE = hexc("#3b2734")
C = dict(
    skin=hexc("#fde9e4"), blush=hexc("#f7a4b4"), hair=hexc("#f6f1fb"), hair_tip=hexc("#ff7f9c"),
    blazer=hexc("#6a6fa6"), blazer2=hexc("#5a5e93"), shirt=hexc("#ffffff"), tie=hexc("#ef4f6f"),
    skirt=hexc("#3f3b5c"), sock=hexc("#ffffff"), shoe=hexc("#3b2734"), white=hexc("#ffffff"),
    iris=hexc("#6e6676"), irisL=hexc("#ff5a72"), irisR=hexc("#5b86ff"), mouth=hexc("#c9566e"), tongue=hexc("#f59aa8"),
)
LW = 4.6   # основная толщина линии (в единицах рисунка при масштабе 1)


class Pen:
    """Обёртка: всё рисуется через трансформацию T, толщина линии масштабируется."""

    def __init__(self, cv, T, lw=LW):
        self.cv, self.T, self.lw = cv, T, lw
        self.k = 0

    def key(self):
        self.k += 1
        return self.k * 13

    def line(self, pts, w=1.0, color=LINE, smooth=True, closed=False):
        P = catmull(pts, closed=closed) if smooth and len(pts) > 2 else np.asarray(pts, float)
        self.cv.stroke(self.T(P), self.lw * w * self.T.scale(), color=color, key=self.key(),
                       taper=(0.06, 0.08), overshoot=0.6, closed=closed)

    def shape(self, pts, color, w=1.0, smooth=True, line=True, clip=None):
        P = catmull(pts, closed=True) if smooth else np.asarray(pts, float)
        self.cv.fill(self.T(P), color, clip=[self.T(c) for c in clip] if clip else None)
        if line:
            self.cv.stroke(self.T(np.vstack([P, P[:2]])), self.lw * w * self.T.scale(), color=LINE, key=self.key(),
                           taper=(0.0, 0.0), overshoot=0, closed=False)
        return P

    def raw(self, P, color, clip=None):
        self.cv.fill(self.T(P), color, clip=[self.T(c) for c in clip] if clip else None)


def M(P):
    P = np.asarray(P, float).copy()
    P[:, 0] *= -1
    return P


# ---------------- голова ----------------

def head(pen, eyes="half", mouth="o", blush=True, tilt=0.0, look=(0, 0), back=True, front=True):
    """Голова. eyes: half | open | xx | closed | worried | side; mouth: o | w | line | grin | open | wavy."""
    T0 = pen.T
    pen.T = Xf(rot=math.radians(tilt), pivot=(0, 110), parent=T0)
    if back:
        # затылок и боковые пряди каре (за лицом), кончики прядей — красные
        outline = path([(-112, 118), (-104, 88), (-120, 40), (-120, -40), (-102, -112), (-58, -152), (0, -166),
                        (58, -152), (102, -112), (120, -40), (122, 40), (108, 88), (116, 120)],
                       [(116, 120), (96, 104), (84, 130)], [(84, 130), (72, 100), (0, 100), (-72, 100), (-86, 130)],
                       [(-86, 130), (-96, 104), (-112, 118)])
        hb = pen.shape(outline, C["hair"], smooth=False)
        pen.raw(np.array([(-130, 98), (130, 98), (130, 140), (-130, 140)]), C["hair_tip"], clip=[hb])
        pen.line(outline, 1.0, smooth=False, closed=True)
        for s_ in (-1, 1):
            pen.line([(s_ * 112, -20), (s_ * 114, 40), (s_ * 104, 90)], 0.6)
    if front:
        face = [(-84, -40), (-86, 20), (-74, 70), (-44, 108), (-8, 126), (2, 127), (40, 110), (72, 72), (86, 22), (84, -40), (0, -80)]
        F = pen.shape(face, C["skin"], line=False)
        pen.line([(-86, 10), (-80, 56), (-56, 94), (-22, 118), (2, 127), (36, 112), (68, 78), (84, 30)], 1.0)
        if blush:
            for s in (-1, 1):
                pen.raw(ellipse(s * 52, 78, 17, 8), C["blush"])
                for j in range(3):
                    x = s * (44 + 7 * j)
                    pen.line([(x - 2, 82), (x + 3, 73)], 0.45, smooth=False)
        for s in (-1, 1):
            eye(pen, s, eyes, look)
        mouth_shape(pen, mouth)
        # чёлка: неровные пряди, пара тонких прядей поперёк глаз (фирменная деталь стиля)
        tips = [(-108, -36), (-100, 8), (-86, -22), (-66, 16), (-52, -18), (-32, 22), (-20, -26), (-4, 36),
                (10, -22), (26, 18), (42, -24), (60, 14), (76, -18), (94, 10), (110, -36)]
        pieces = []
        for a, b in zip(tips, tips[1:]):
            down = b[1] > a[1]
            m = ((a[0] + b[0]) / 2 + (-3 if down else 3), (a[1] + b[1]) / 2 + (-5 if down else 6))
            pieces.append([a, m, b])
        edge = path(*pieces)
        top = catmull([(110, -36), (112, -90), (82, -140), (20, -160), (-48, -152), (-96, -112), (-108, -36)])
        bang = np.vstack([edge, top[1:]])
        pen.shape(bang, C["hair"], smooth=False)
        pen.line([(-30, -110), (-28, -66), (-34, -30)], 0.6)
        pen.line([(44, -116), (46, -72), (40, -36)], 0.6)
        pen.line([(-64, -40), (-52, 26), (-46, 60)], 0.55)
        pen.line([(54, -40), (46, 24), (42, 52)], 0.55)
        # ахоге-завиток
        aho = catmull([(-4, -160), (-12, -184), (2, -200), (16, -192), (8, -178)])
        pen.line(aho, 1.0, smooth=False)
        # заколка-крестик
        for a in (0.7, -0.7):
            c, s = math.cos(a), math.sin(a)
            pen.line([(-62 - 12 * c, -88 - 12 * s), (-62 + 12 * c, -88 + 12 * s)], 1.6, color=C["tie"], smooth=False)
    pen.T = T0


def eye(pen, s, kind, look):
    m = (lambda P: np.asarray(P, float)) if s < 0 else M
    lx, ly = look
    if kind == "xx":       # ><
        pen.line(m([(-58, 28), (-36, 42), (-58, 54)]), 1.1, smooth=False)
        return
    if kind == "closed":
        pen.line(m([(-60, 44), (-38, 50), (-16, 44)]), 1.1)
        return
    if kind == "happy":
        pen.line(m([(-58, 48), (-38, 34), (-18, 48)]), 1.1)
        return
    top = {"half": 28, "open": 18, "worried": 22, "side": 28}[kind]
    white = m([(-62, top + 2), (-60, 58), (-40, 68), (-18, 62), (-14, top - 2), (-38, top - 4)])
    pen.raw(catmull(white, closed=True), C["white"])
    ix, iy = -38 + lx, 46 + ly
    if s > 0:
        ix = -38 - lx
    ir = m(ellipse(ix, iy + 2, 15, 20))
    pen.raw(ir, C["iris"], clip=[catmull(white, closed=True)])
    cres = m(catmull([(ix - 11, iy + 4), (ix, iy + 9), (ix + 11, iy + 4), (ix + 9, iy + 13), (ix, iy + 17), (ix - 9, iy + 13)], closed=True))
    pen.raw(cres, C["irisL"] if s < 0 else C["irisR"], clip=[catmull(white, closed=True)])
    pen.raw(m(ellipse(ix + 4, iy - 7, 4.5, 4.5)), C["white"], clip=[catmull(white, closed=True)])
    if kind == "worried":
        pen.line(m([(-62, top + 2), (-40, top - 6), (-14, top - 10)]), 1.3)
        pen.line(m([(-60, -6), (-40, -14), (-22, -24)]), 0.8)
    else:
        pen.line(m([(-66, top + 4), (-40, top - 3), (-12, top - 4)]), 1.7)
        pen.line(m([(-64, top + 4), (-70, top)]), 0.8, smooth=False)
    pen.line(m([(-50, 64), (-34, 66)]), 0.7, smooth=False)


def mouth_shape(pen, kind):
    y = 100
    if kind == "o":
        pen.shape(ellipse(4, y, 6, 7), C["mouth"], 0.8)
    elif kind == "open":
        pen.shape([(-8, y - 6), (12, y - 6), (10, y + 8), (0, y + 12), (-8, y + 6)], C["mouth"], 0.8)
        pen.raw(ellipse(2, y + 7, 6, 3.5), C["tongue"])
    elif kind == "w":
        pen.line([(-10, y - 2), (-5, y + 3), (1, y - 1), (7, y + 3), (12, y - 2)], 0.75, smooth=False)
    elif kind == "line":
        pen.line([(-6, y), (10, y + 1)], 0.75, smooth=False)
    elif kind == "grin":
        pen.line([(-12, y - 3), (2, y + 5), (16, y - 3)], 0.8)
    elif kind == "wavy":
        pen.line([(-10, y), (-4, y - 3), (2, y + 1), (8, y - 3), (14, y)], 0.75)
    elif kind == "pout":
        pen.line([(-2, y - 6), (6, y - 2), (-2, y + 2), (6, y + 6)], 0.75, smooth=False)


# ---------------- тело ----------------

def offset(P, d):
    P = np.asarray(P, float)
    t = np.gradient(P, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
    n = np.stack([-t[:, 1], t[:, 0]], 1)
    return P + n * (np.asarray(d)[:, None] if np.ndim(d) else d)


def torso(pen, hem=440, open_=True):
    """Пиджак-оверсайз, воротник рубашки, галстук."""
    neck = [(-18, 104), (-20, 150), (0, 160), (20, 150), (18, 104)]
    pen.shape(neck, C["skin"], smooth=True)
    body = [(-24, 146), (-62, 160), (-80, 176), (-86, 260), (-92, 360), (-98, hem), (98, hem), (92, 360), (86, 260),
            (80, 176), (62, 160), (24, 146)]
    pen.shape(body, C["blazer"], smooth=True)
    # рубашка в вырезе
    pen.shape([(-24, 148), (24, 148), (6, 240), (-6, 240)], C["shirt"], smooth=False)
    pen.shape([(-24, 146), (-30, 164), (-6, 178), (0, 160)], C["shirt"], 0.8, smooth=False)
    pen.shape([(24, 146), (30, 164), (6, 178), (0, 160)], C["shirt"], 0.8, smooth=False)
    pen.shape([(-6, 168), (6, 168), (9, 222), (0, 232), (-9, 222)], C["tie"], 0.8, smooth=False)
    pen.line([(-24, 150), (-8, 240), (0, hem)], 0.8)
    pen.line([(24, 150), (8, 240)], 0.8)
    for by in (300, 370):
        pen.shape(ellipse(-12, by, 4.5, 4.5), C["white"], 0.6)


def sleeve(pen, pts, w0=58, w1=62, hand=None, hand_ang=0.0, cuff=True, skip=0.18):
    """Длинный рукав по осевой линии; кисть (кулачок) выглядывает из манжеты."""
    Cc = catmull(pts, step=3)
    n = len(Cc)
    w = np.linspace(w0, w1, n) / 2
    L, R = offset(Cc, w), offset(Cc, -w)
    end, d = Cc[-1], Cc[-1] - Cc[-4]
    d /= np.linalg.norm(d) + 1e-9
    nrm = np.array([-d[1], d[0]])
    cap = [end + nrm * w[-1] * math.cos(a) + d * w[-1] * 0.25 * math.sin(a) for a in np.linspace(0, math.pi, 9)]
    if hand:
        fist(pen, end + d * 6, math.atan2(d[1], d[0]) + hand_ang, hand)
    poly = np.vstack([L, np.array(cap)[::-1] if False else np.array(cap), R[::-1]])
    pen.raw(poly, C["blazer"])
    k0 = int(n * skip)          # начало рукава сливается с плечом — без контура
    pen.line(L[k0:], 1.0, smooth=False)
    pen.line(R[k0:], 1.0, smooth=False)
    pen.line(np.array(cap), 1.0, smooth=False)
    if cuff:
        for k in (-0.5, 0.0, 0.5):
            a = end - d * 4 + nrm * w[-1] * k
            pen.line([a, a - d * 22], 0.55, smooth=False)
        band = [end - d * 26 + nrm * w[-1] * s for s in (-1, 0, 1)]
        pen.line([band[0], band[1] - d * 2, band[2]], 0.6)
    return end, d


def fist(pen, at, ang, kind="fist"):
    """Маленькая кисть, выглядывающая из рукава (как в клипе)."""
    c, s = math.cos(ang), math.sin(ang)

    def X(P):
        P = np.asarray(P, float)
        return np.stack([c * P[:, 0] - s * P[:, 1] + at[0], s * P[:, 0] + c * P[:, 1] + at[1]], 1)
    if kind == "fist":
        pen.shape(X([(-4, -16), (14, -16), (24, -6), (24, 8), (14, 16), (-4, 16)]), C["skin"], 0.9)
        pen.line(X([(14, -6), (22, -4)]), 0.5, smooth=False)
        pen.line(X([(14, 4), (22, 5)]), 0.5, smooth=False)
    elif kind == "open":
        pen.shape(X([(-4, -18), (20, -20), (34, -14), (36, -4), (30, 6), (36, 12), (24, 18), (-4, 16)]), C["skin"], 0.9)
        pen.line(X([(20, -10), (32, -8)]), 0.5, smooth=False)
        pen.line(X([(18, 0), (30, 0)]), 0.5, smooth=False)
    elif kind == "pinch":
        pen.shape(X([(-4, -14), (16, -16), (26, -8), (22, 4), (28, 12), (14, 16), (-4, 14)]), C["skin"], 0.9)
        pen.line(X([(14, -6), (22, -4)]), 0.5, smooth=False)


def legs(pen, left=(0, 0), right=(0, 0), skirt_y=420):
    """Юбка и ноги. left/right — (угол в градусах, сгиб колена) для каждой ноги."""
    pen.shape([(-96, skirt_y), (96, skirt_y), (112, skirt_y + 110), (-112, skirt_y + 110)], C["skirt"], smooth=False)
    for x in (-60, -20, 20, 60):
        pen.line([(x, skirt_y + 10), (x * 1.18, skirt_y + 106)], 0.55, smooth=False)
    for side, (ang, knee) in ((-1, left), (1, right)):
        hip = np.array([side * 42.0, skirt_y + 100])
        a = math.radians(ang)
        kpt = hip + 180 * np.array([math.sin(a), math.cos(a)])
        a2 = a + math.radians(knee)
        foot = kpt + 190 * np.array([math.sin(a2), math.cos(a2)])
        thigh = [hip + (-16, 0), kpt + (-12, 0), kpt + (12, 0), hip + (16, 0)]
        shin_w = 11
        d = (foot - kpt) / np.linalg.norm(foot - kpt)
        nn = np.array([-d[1], d[0]])
        pen.shape(np.array(thigh), C["skin"], smooth=False)
        sock = [kpt + nn * 12 + d * 40, foot + nn * shin_w, foot - nn * shin_w, kpt - nn * 12 + d * 40]
        pen.shape(np.array([kpt + nn * 12, kpt + nn * 12 + d * 40, kpt - nn * 12 + d * 40, kpt - nn * 12]), C["skin"], smooth=False)
        pen.shape(np.array(sock), C["sock"], smooth=False)
        shoe = [foot + nn * 13 - d * 4, foot + nn * 13 + d * 14 + nn * 10 * side * 0, foot + d * 18 - nn * side * 26,
                foot - nn * 13 + d * 10, foot - nn * 13 - d * 4]
        pen.shape(np.array(shoe), C["shoe"], 0.8)
