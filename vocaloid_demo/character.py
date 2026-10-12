"""Персонаж: «Нотка» — девочка-вокалоид с хвостиками и гарнитурой.

Всё описано контрольными точками в локальных координатах головы
(0,0 — между глаз, единица ≈ пиксель кадра 1280×720).
"""
import math
import numpy as np
from ink import path, catmull, ellipse, hexc, Xf, INK, PENCIL_BLUE, resample, resample_n, plen

X0, Y0 = 420, 300
TIPS = [(-108, -22), (-95, 24), (-82, -14), (-65, 7), (-51, -24), (-34, 5), (-20, -30), (-3, 31),
        (10, -30), (27, 3), (44, -24), (62, 8), (79, -12), (97, 27), (110, -22)]           # где на листе голова
NECK_PIVOT = (0, 140)

C = dict(
    skin=hexc("#fbe0cf"), skin_sh=hexc("#e9b4ae"), skin_sh2=hexc("#d8c0de"),
    hair=hexc("#45bfb6"), hair_sh=hexc("#2c8796"), hair_hi=hexc("#e9fffb"),
    shirt=hexc("#e3e3e6"), shirt_sh=hexc("#b5b6c8"),
    tie=hexc("#3fb6ad"), tie_sh=hexc("#2a7f8c"),
    dark=hexc("#4b4e5a"), glow=hexc("#78f0e4"), pink=hexc("#f2709e"),
    iris=hexc("#35b0aa"), iris_d=hexc("#1c5868"), mouth=hexc("#a94456"), tongue=hexc("#ef8c96"),
    blush=hexc("#f39aa8"), white=np.array([0.985, 0.98, 0.965]), red=hexc("#d9474f"),
)

# небольшая «ручная» асимметрия: правый глаз не зеркальная копия левого
_r = np.random.default_rng(42)
ASYM = _r.normal(0, 1.4, (16, 2))


def mirror(P, asym=False):
    P = np.asarray(P, float).copy()
    P[:, 0] *= -1
    if asym:  # плавная, а не точечная асимметрия
        x, y = P[:, 0].copy(), P[:, 1].copy()
        P[:, 0] += 1.4 * np.sin(0.031 * y + 0.7)
        P[:, 1] += 1.1 * np.sin(0.027 * x + 1.9)
    return P


class Part:
    def __init__(self, name, order):
        self.name, self.order = name, order
        self.occ, self.fills, self.strokes, self.hatches = [], [], [], []

    def S(self, pts, w=2.0, **kw):
        self.strokes.append((np.asarray(pts, float), w, kw))

    def F(self, poly, color, **kw):
        self.fills.append((np.asarray(poly, float), color, kw))

    def O(self, poly):
        self.occ.append(np.asarray(poly, float))

    def Hh(self, poly, angle, gap, **kw):
        self.hatches.append((np.asarray(poly, float), angle, gap, kw))

    def xf(self, T):
        self.occ = [T(p) for p in self.occ]
        self.fills = [(T(p), c, {**kw, **({"clip": [T(q) for q in kw["clip"]]} if "clip" in kw else {}),
                                 **({"holes": [T(q) for q in kw["holes"]]} if "holes" in kw else {})})
                      for p, c, kw in self.fills]
        self.strokes = [(T(p), w, kw) for p, w, kw in self.strokes]
        self.hatches = [(T(p), a, g, kw) for p, a, g, kw in self.hatches]
        return self


def offset_curve(P, d):
    P = np.asarray(P, float)
    t = np.gradient(P, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
    n = np.stack([-t[:, 1], t[:, 0]], 1)
    return P + n * (d[:, None] if np.ndim(d) else d)


def twintail(side, phase, amp):
    s = side
    base = np.array([s * 118, -116])
    ctrl = np.array([[0, 0], [s * 52, 40], [s * 90, 150], [s * 104, 285], [s * 98, 420], [s * 86, 560]], float)
    sway = np.array([amp * math.sin(phase - i * 0.75) * (i / 5) ** 1.3 for i in range(6)])
    ctrl[:, 0] += sway * s * 0.6 + sway * 0.4
    ctrl[:, 1] += np.abs(sway) * 0.15
    C_ = catmull(ctrl + base, step=3)
    n = len(C_)
    u = np.linspace(0, 1, n)
    hw = np.interp(u, [0, 0.08, 0.3, 0.6, 1.0], [16, 30, 47, 50, 44])
    left = offset_curve(C_, hw)
    right = offset_curve(C_, -hw)
    poly = np.vstack([left, right[::-1]])
    inner = [offset_curve(C_, hw * k)[int(n * a):int(n * b)] for k, a, b in
             [(0.45, 0.12, 0.75), (-0.15, 0.25, 1.0), (-0.6, 0.08, 0.55), (0.1, 0.5, 1.0)]]
    shadow = np.vstack([offset_curve(C_, hw * (-0.25 * s))[int(n * 0.05):], right[::-1][: n - int(n * 0.05)] if s > 0 else left[::-1][: n - int(n * 0.05)]])
    return poly, left, right, inner, C_, shadow


def build(pose):
    rot = math.radians(pose.get("rot", 0.0))
    head = Xf(tx=X0 + pose.get("dx", 0), ty=Y0 + pose.get("dy", 0), rot=rot, pivot=NECK_PIVOT)
    body = Xf(tx=X0, ty=Y0 + pose.get("breathe", 0) * 0.5, pivot=(0, 215), sx=0.84, sy=1 + pose.get("breathe", 0) * 0.004)
    parts = []

    # ---------- хвостики (самое заднее) ----------
    for side, nm in ((-1, "tailL"), (1, "tailR")):
        p = Part(nm, 8)
        poly, le, ri, inner, cl, sh = twintail(side, pose.get("tail_phase", 0) + (0.4 if side > 0 else 0), pose.get("tail_amp", 0))
        p.O(poly)
        p.F(poly, C["hair"], alpha=0.8, key=11 + side)
        p.F(sh, C["hair_sh"], alpha=0.45, key=21 + side, edge=0.6)
        p.S(le, 2.4)
        p.S(ri, 2.4)
        for k, q in enumerate(inner):
            if len(q) > 5:
                p.S(q, 1.1, alpha=0.7, taper=(0.3, 0.5))
        p.Hh(sh[: len(sh) // 2 + 20], math.radians(60 * side), 7, alpha=0.28)
        parts.append(p.xf(head))

    # ---------- задние волосы ----------
    p = Part("backhair", 6)
    back = [(-128, 185), (-134, 60), (-128, -40), (-112, -118), (-66, -166), (0, -180), (66, -166), (112, -118),
            (128, -40), (134, 60), (128, 185), (92, 196), (0, 170), (-92, 196)]
    bpoly = catmull(back, closed=True)
    p.O(bpoly)
    p.F(bpoly, C["hair"], alpha=0.8, key=31)
    p.F(catmull([(-125, 185), (-120, 40), (-60, 30), (60, 30), (120, 40), (125, 185), (0, 170)], closed=True), C["hair_sh"], alpha=0.55, key=32)
    p.S(catmull([(-128, 188), (-134, 60), (-130, -30)]), 2.3)
    p.S(catmull([(128, 188), (134, 60), (130, -30)]), 2.3)
    p.S(catmull([(-128, 188), (-110, 194), (-94, 196)]), 1.6)
    p.S(catmull([(128, 188), (110, 194), (94, 196)]), 1.6)
    parts.append(p.xf(head))

    # ---------- шея ----------
    p = Part("neck", 9)
    neck = [(-30, 120), (-33, 170), (-37, 212), (0, 228), (37, 212), (33, 170), (30, 120)]
    nP = catmull(neck, closed=True)
    p.O(nP)
    p.F(nP, C["skin"], alpha=0.75, key=61)
    p.F(catmull([(-31, 120), (-32, 168), (0, 182), (32, 168), (31, 120)], closed=True), C["skin_sh"], alpha=0.6, key=62)
    p.Hh(catmull([(-31, 130), (-32, 165), (0, 178), (32, 165), (31, 130)], closed=True), math.radians(-35), 6, alpha=0.3)
    p.S(catmull([(-30, 122), (-33, 170), (-38, 210)]), 2.0)
    p.S(catmull([(30, 122), (33, 170), (38, 210)]), 2.0)
    parts.append(p.xf(Xf(rot=rot * 0.35, pivot=(0, 210), tx=X0, ty=Y0)))

    # ---------- тело ----------
    p = Part("body", 9)
    armL = [(-118, 236), (-152, 242), (-194, 268), (-214, 322), (-222, 440), (-142, 440), (-144, 340), (-132, 280)]
    shirt = [(-36, 200), (-74, 226), (-118, 236), (-132, 280), (-144, 340), (-142, 440), (142, 440), (144, 340),
             (132, 280), (118, 236), (74, 226), (36, 200), (30, 214), (0, 236), (-30, 214)]
    for a in (armL, mirror(armL)):
        P = catmull(a, closed=True)
        p.O(P)
        p.F(P, C["skin"], alpha=0.75, key=41)
    sP = catmull(shirt, closed=True)
    p.O(sP)
    p.F(sP, C["shirt"], alpha=0.8, key=42)
    p.F(catmull([(-120, 300), (-60, 330), (0, 345), (60, 330), (120, 300), (142, 440), (-142, 440)], closed=True), C["shirt_sh"], alpha=0.35, key=43)
    p.F(catmull([(-148, 250), (-140, 300), (-150, 440), (-172, 440), (-168, 300)], closed=True), C["skin_sh"], alpha=0.45, key=44)
    p.F(catmull([(148, 250), (140, 300), (150, 440), (172, 440), (168, 300)], closed=True), C["skin_sh"], alpha=0.45, key=45)
    # плечи и руки
    p.S(catmull([(-40, 205), (-90, 228), (-152, 240), (-195, 268), (-214, 322), (-222, 445)]), 2.6)
    p.S(catmull([(40, 205), (90, 228), (152, 240), (195, 268), (214, 322), (222, 445)]), 2.6)
    p.S(catmull([(-118, 238), (-131, 282), (-144, 340), (-142, 445)]), 2.0)
    p.S(catmull([(118, 238), (131, 282), (144, 340), (142, 445)]), 2.0)
    # складки рубашки
    p.S(catmull([(-100, 330), (-80, 360), (-74, 400)]), 1.1, alpha=0.6)
    p.S(catmull([(96, 320), (84, 350), (86, 390)]), 1.1, alpha=0.6)
    p.S(catmull([(-40, 300), (-34, 330)]), 1.0, alpha=0.5)
    # «тату» с номером на плече (как у вокалоидов)
    parts.append(p.xf(body))
    p.tattoo = (body(np.array([[160, 286]]))[0], "07")

    # воротник и галстук
    p = Part("collar", 10)
    colL = [(-38, 198), (-78, 232), (-56, 262), (-6, 246), (0, 234)]
    for k, cpts in enumerate((colL, mirror(colL, True))):
        P = catmull(cpts, closed=True)
        p.O(P)
        p.F(P, C["shirt"], alpha=0.7, key=51 + k)
        p.S(P, 2.0, closed=True)
    p.F(catmull([(-30, 236), (0, 252), (30, 236), (32, 252), (0, 268), (-32, 252)], closed=True), C["shirt_sh"], alpha=0.4, key=53)
    knot = [(-12, 238), (12, 238), (9, 260), (-9, 260)]
    tie = [(-9, 258), (-15, 330), (-17, 400), (0, 440), (17, 400), (15, 330), (9, 258)]
    tP = catmull(tie, closed=True)
    p.O(tP)
    p.F(tP, C["tie"], alpha=0.85, key=54)
    p.F(catmull([(0, 262), (8, 330), (12, 400), (0, 438), (2, 330)], closed=True), C["tie_sh"], alpha=0.5, key=55)
    p.S(tP, 2.0)
    kP = catmull(knot, closed=True)
    p.O(kP)
    p.F(kP, C["tie"], alpha=0.95, key=56)
    p.S(kP, 2.0, closed=True)
    p.S(catmull([(-3, 290), (-5, 330)]), 0.9, alpha=0.6)
    parts.append(p.xf(body))

    # ---------- лицо ----------
    p = Part("face", 1)
    jaw = [(-99, -30), (-100, 5), (-95, 45), (-81, 82), (-59, 114), (-31, 138), (-6, 149), (3, 150),
           (31, 137), (59, 113), (81, 81), (95, 44), (100, 4), (99, -30)]
    face = catmull(jaw + [(70, -110), (0, -125), (-70, -110)], closed=True)
    p.O(face)
    p.F(face, C["skin"], alpha=0.7, key=71)
    # тень от чёлки на лбу
    sh = [(-99, -40)] + [(x + (2 if x > 0 else -2), y + 10) for x, y in TIPS[1:-1]] + [(99, -40)]
    p.F(catmull(sh, closed=True), C["skin_sh2"], alpha=0.55, key=72)
    p.S(catmull(jaw[1:-1]), 2.5, taper=(0.1, 0.1))
    # румянец
    for sx in (-1, 1):
        p.F(ellipse(sx * 58, 92, 20, 9, 0), C["blush"], alpha=0.45, key=73 + sx, edge=0.1, tex=0.5, blur=4)
        for k in range(3):
            x = sx * (48 + k * 9)
            p.S([(x - 3, 96), (x + 3, 87)], 1.1, color=hexc("#c9566e"), alpha=0.6, overshoot=0)
    # нос
    p.S(catmull([(4, 80), (1, 86), (-2, 88)]), 1.3, alpha=0.8)
    parts.append(p.xf(head))

    # ---------- глаза ----------
    blink = pose.get("blink", 0)  # 0 открыт, 1 полуприкрыт, 2 закрыт, 3 «счастливые ^^»
    p = Part("eyes", 2)
    for side in (-1, 1):
        mir = (lambda P: np.asarray(P, float)) if side < 0 else (lambda P: mirror(P, True))
        if blink in (2, 3):
            if blink == 2:
                p.S(mir(catmull([(-82, 50), (-66, 58), (-48, 60), (-28, 56), (-18, 50)])), 3.6, taper=(0.2, 0.4))
            else:
                p.S(mir(catmull([(-80, 54), (-64, 36), (-48, 30), (-32, 36), (-18, 52)])), 3.8, taper=(0.25, 0.35))
            p.S(mir(catmull([(-76, -24), (-52, -32), (-26, -26)])), 1.6, taper=(0.2, 0.4))
            continue
        lid = 0 if blink == 0 else 22
        white = mir(catmull([(-80, 26 + lid), (-76, 54), (-60, 72), (-40, 76), (-24, 68), (-17, 44), (-18, 22 + lid),
                             (-34, 12 + lid), (-54, 9 + lid), (-70, 14 + lid)], closed=True))
        p.F(white, C["white"], alpha=0.9, mode="normal", shift=False, edge=0, tex=0.1, key=81 + side)
        iris = mir(ellipse(-47, 46, 21, 29, 0.03))
        p.F(iris, C["iris"], alpha=0.95, clip=[white], key=82 + side, shift=False, wobble=0.5)
        p.F(mir(ellipse(-47, 30, 22, 16)), C["iris_d"], alpha=0.55, clip=[white, iris] if False else [iris], key=83 + side, shift=False)
        p.F(mir(ellipse(-47, 44, 9, 13)), C["iris_d"], alpha=0.75, clip=[white], key=84 + side, shift=False)
        p.F(mir(catmull([(-62, 60), (-47, 72), (-32, 60), (-47, 66)], closed=True)), hexc("#9cf0e0"), alpha=0.6, clip=[iris], key=85 + side)
        p.S(iris, 1.4, closed=True, alpha=0.85)
        # тень века на белке
        p.F(mir(catmull([(-80, 26 + lid), (-54, 10 + lid), (-18, 22 + lid), (-30, 30 + lid), (-54, 24 + lid), (-74, 34 + lid)], closed=True)),
            C["skin_sh2"], alpha=0.5, clip=[white], key=86 + side)
        # блики
        p.F(mir(ellipse(-56, 30 + lid * 0.6, 7.5, 9.5, 0.4)), C["white"], alpha=0.97, mode="normal", shift=False, edge=0, tex=0.05, key=87 + side, wobble=0.4)
        p.F(mir(ellipse(-37, 58, 3.2, 3.2)), C["white"], alpha=0.95, mode="normal", shift=False, edge=0, tex=0.05, key=88 + side, wobble=0.3)
        # ресницы
        p.S(mir(catmull([(-16, 24 + lid), (-30, 12 + lid), (-52, 7 + lid), (-72, 13 + lid), (-84, 26 + lid)])), 5.2, taper=(0.35, 0.12))
        p.S(mir(catmull([(-80, 20 + lid), (-89, 15 + lid), (-93, 17 + lid)])), 2.4, taper=(0.1, 0.6), overshoot=0)
        p.S(mir(catmull([(-82, 30 + lid), (-90, 31 + lid)])), 1.8, taper=(0.1, 0.7), overshoot=0)
        p.S(mir(catmull([(-71, 66), (-60, 73), (-50, 75)])), 1.4, taper=(0.4, 0.4))
        p.S(mir(catmull([(-30, 71), (-24, 66)])), 1.2)
        # брови
        p.S(mir(catmull([(-78, -24), (-54, -33), (-26, -27)])), 1.7, taper=(0.15, 0.5))
    parts.append(p.xf(head))

    # ---------- рот ----------
    p = Part("mouth", 3)
    vowel, op = pose.get("mouth", (None, 0.0))
    smile = pose.get("smile", 1.0)
    if vowel is None or op < 0.08 or vowel == "m":
        p.S(catmull([(-13, 112 - 3 * smile), (-4, 116), (5, 116), (13, 111 - 3 * smile)]), 1.8, taper=(0.3, 0.3))
    else:
        wdt = {"a": 13, "o": 9, "u": 6, "i": 15, "e": 13, "y": 11}[vowel]
        hgt = {"a": 18, "o": 17, "u": 11, "i": 8, "e": 13, "y": 10}[vowel] * (0.35 + 0.65 * op)
        top = 109 - 1.5
        M = catmull([(-wdt, top), (-wdt * 0.4, top + 1.5), (wdt * 0.4, top + 1.5), (wdt, top),
                     (wdt * 0.75, top + hgt * 0.75), (0, top + hgt), (-wdt * 0.75, top + hgt * 0.75)], closed=True)
        p.F(M, C["mouth"], alpha=0.9, mode="normal", shift=False, edge=0.2, key=91)
        if hgt > 9:
            p.F(ellipse(0, top + hgt * 0.78, wdt * 0.6, hgt * 0.28), C["tongue"], alpha=0.9, mode="normal", clip=[M], shift=False, key=92)
        if vowel in ("a", "e", "i") and hgt > 8:
            p.F(catmull([(-wdt * 0.8, top + 1), (wdt * 0.8, top + 1), (wdt * 0.6, top + 4), (-wdt * 0.6, top + 4)], closed=True),
                C["white"], alpha=0.95, mode="normal", clip=[M], shift=False, edge=0, key=93)
        p.S(M, 1.7, closed=True)
    parts.append(p.xf(head))

    # ---------- передние пряди по бокам лица ----------
    for side in (-1, 1):
        p = Part("lock", 5)
        lock_o = [(-110, -70), (-118, 30), (-118, 140), (-112, 250), (-104, 330)]
        lock_i = [(-104, 330), (-96, 250), (-90, 150), (-86, 60), (-90, -20)]
        mo = (lambda P: np.asarray(P, float)) if side < 0 else (lambda P: mirror(P, True))
        o = mo(catmull(lock_o))
        i_ = mo(catmull(lock_i))
        P = np.vstack([o, i_])
        sway = pose.get("tail_amp", 0) * 0.15 * math.sin(pose.get("tail_phase", 0) - 1.0)
        bend = np.clip((P[:, 1] - 80) / 250, 0, 1) ** 1.5 * sway
        P = P + np.stack([bend, 0 * bend], 1)
        o, i_ = P[:len(o)], P[len(o):]
        p.O(P)
        p.F(P, C["hair"], alpha=0.85, key=101 + side)
        m = 160
        oo, ii = resample_n(o, m), resample_n(i_[::-1], m)
        p.F(np.vstack([ii, (ii * 0.62 + oo * 0.38)[::-1]]), C["hair_sh"], alpha=0.4, key=103 + side)
        p.S(o, 2.3)
        p.S(i_, 2.0)
        mid = (oo * 0.5 + ii * 0.5)[m // 5:int(m * 0.85)]
        p.S(mid, 1.0, alpha=0.6, taper=(0.3, 0.6))
        parts.append(p.xf(head))

    # ---------- чёлка ----------
    p = Part("bangs", 4)
    tips = TIPS
    pieces = []
    for a, b in zip(tips, tips[1:]):
        m = ((a[0] + b[0]) / 2 + (2.5 if b[1] > a[1] else -2.5), (a[1] + b[1]) / 2 + (-4 if b[1] > a[1] else 4))
        pieces.append([a, m, b])
    edge = path(*pieces)
    top = catmull([(110, -30), (114, -80), (100, -128), (60, -162), (0, -176), (-60, -162), (-100, -128), (-114, -80), (-108, -30)])
    bang = np.vstack([edge, top[1:]])
    p.O(bang)
    p.F(bang, C["hair"], alpha=0.85, key=111)
    p.F(np.vstack([edge, catmull([(110, -30), (60, -50), (0, -58), (-60, -50), (-108, -30)])[1:]]), C["hair_sh"], alpha=0.35, key=112)
    # блик-полоска на волосах
    hl = []
    for k in range(9):
        x = -90 + k * 22.5
        y = -118 + 0.004 * x * x
        hl.append((x, y + (0 if k % 2 else 9)))
    hl_poly = np.vstack([catmull(hl), catmull([(x, y + 13 + (k % 2) * 4) for k, (x, y) in enumerate(hl)])[::-1]])
    p.F(hl_poly, C["hair_hi"], alpha=0.55, mode="normal", key=113, edge=0.0, tex=0.5, clip=[bang])
    p.S(edge, 2.3, taper=(0.05, 0.05))
    p.S(top, 2.5, taper=(0.1, 0.1))
    for a, b in zip(tips[1::2], tips[2::2]):
        p.S(catmull([(b[0] + 3, b[1] - 6), (b[0] + 6, b[1] - 30), (b[0] + 4, b[1] - 60)]), 1.0, alpha=0.6, taper=(0.5, 0.2))
    for x in (-70, -25, 30, 70):
        p.S(catmull([(x, -150 + abs(x) * 0.3), (x * 0.9, -110), (x * 0.95, -80)]), 1.0, alpha=0.5, taper=(0.5, 0.5))
    sway = 4 * math.sin(pose.get("tail_phase", 0) * 1.3)
    aho = path([(8, -172), (16 + sway * 0.3, -196), (34 + sway, -212), (52 + sway, -206)],
               [(52 + sway, -206), (34 + sway, -203), (22 + sway * 0.5, -190), (16, -173)])
    p.O(aho)
    p.F(aho, C["hair"], alpha=0.85, key=114)
    p.S(aho[: len(aho) // 2 + 2], 2.0, taper=(0.1, 0.05))
    p.S(aho[len(aho) // 2:], 1.8, taper=(0.05, 0.4))
    p.Hh(catmull([(-100, -40), (-60, -50), (0, -56), (60, -50), (100, -40), (80, -22), (46, -34), (12, -40), (-18, -40), (-48, -36), (-80, -24)], closed=True),
         math.radians(70), 6, alpha=0.3)
    parts.append(p.xf(head))

    # ---------- резинки хвостиков ----------
    p = Part("ties", 7)
    for side in (-1, 1):
        cx, cy = side * 120, -114
        box = ellipse(cx, cy, 15, 21, side * 0.6, 16)
        box = box + _r.normal(0, 0.3, box.shape)
        p.O(box)
        p.F(box, C["dark"], alpha=0.9, key=121 + side)
        p.F(ellipse(cx, cy, 6, 9, side * 0.6), C["pink"], alpha=0.9, mode="normal", key=123 + side)
        p.S(box, 2.2, closed=True)
    parts.append(p.xf(head))

    # ---------- гарнитура ----------
    p = Part("headset", 7)
    glow = pose.get("glow", 0.0)
    for side in (-1, 1):
        cup = ellipse(side * 106, 22, 21, 33, side * 0.12, 32)
        p.O(cup)
        p.F(cup, C["dark"], alpha=0.85, key=131 + side)
        ring = ellipse(side * 106, 22, 12, 21, side * 0.12, 24)
        p.F(ring, C["glow"] * (1 - 0.25 * glow) + glow * 0.25, alpha=0.9, mode="normal", key=133 + side)
        p.S(cup, 2.3, closed=True)
        p.S(ring, 1.2, closed=True, alpha=0.7)
    arm = catmull([(-110, 50), (-96, 92), (-70, 116), (-36, 124)])
    p.S(arm, 3.2, taper=(0.05, 0.05))
    mic = ellipse(-30, 124, 7, 5, 0.3)
    p.O(mic)
    p.F(mic, C["dark"], alpha=0.9, mode="normal", key=135)
    p.S(mic, 1.6, closed=True)
    parts.append(p.xf(head))

    return parts


def construction(pose):
    """Синие карандашные построения (не до конца стёртые)."""
    rot = math.radians(pose.get("rot", 0.0))
    head = Xf(tx=X0 + pose.get("dx", 0), ty=Y0 + pose.get("dy", 0), rot=rot, pivot=NECK_PIVOT)
    L = [
        ellipse(4, -22, 122, 128, 0.0, 64)[list(range(64)) + [0, 1, 2]],
        catmull([(2, -150), (4, -40), (2, 60), (0, 160)]),
        catmull([(-130, 38), (0, 42), (132, 36)]),
        catmull([(-110, 50), (-60, 128), (0, 156), (62, 126), (108, 46)]),
    ]
    L = [head(q) for q in L]
    body = Xf(tx=X0, ty=Y0)
    L += [body(catmull([(-230, 252), (0, 214), (232, 250)])),
          body(catmull([(0, 170), (0, 260), (2, 430)]))]
    for side in (-1, 1):
        L.append(head(catmull([(side * 118, -116), (side * 175, -60), (side * 215, 80), (side * 222, 260), (side * 214, 420)])))
    return L
