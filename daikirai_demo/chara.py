"""Оригинальный персонаж «Рэй» (Rei): короткое растрёпанное каре, пиджак с кантом,
гетерохромия красный/синий, прищепка-крестик, клык. Стиль — плоский, толстый контур.

Координаты: начало — между глаз; шея крепится в (0, 128).
"""
import math
import numpy as np
from flat import catmull, path, ellipse, capsule, hexc, Xf, LINE, offset_curve

C = dict(
    skin=hexc("#ffe6dc"), skin_sh=hexc("#f2b4b8"), blush=hexc("#ff7d9a"),
    hair=hexc("#efeaf8"), hair_sh=hexc("#b8a9e0"), hair_in=hexc("#ff2e55"), hair_in_sh=hexc("#c0164a"),
    blazer=hexc("#2a2c63"), blazer_sh=hexc("#1b1c45"), pipe=hexc("#3d63ff"),
    shirt=hexc("#f6f3fa"), shirt_sh=hexc("#c7c0de"), tie=hexc("#ff2e55"), tie_sh=hexc("#b81445"),
    gold=hexc("#ffc94a"), white=hexc("#ffffff"), eye_white=hexc("#fffafc"),
    irisL=hexc("#ff2a48"), irisL_d=hexc("#8f0a2c"), irisR=hexc("#2f5cff"), irisR_d=hexc("#131f86"),
    mouth=hexc("#5c0b2b"), tongue=hexc("#ff6f8d"), clip=hexc("#ff2e55"), clip2=hexc("#3d63ff"),
)
NECK = (0.0, 128.0)
BODY_S = 0.84
BODY_PIVOT = (0.0, 150.0)


def to_body(p):
    """Точка из координат персонажа (как у головы) в координаты тела."""
    return ((p[0] - BODY_PIVOT[0]) / BODY_S + BODY_PIVOT[0], (p[1] - BODY_PIVOT[1]) / BODY_S + BODY_PIVOT[1])
W_OUT = 3.4   # толщина основного контура (в единицах персонажа)


def mir(P):
    P = np.asarray(P, float).copy()
    P[:, 0] *= -1
    return P


def ik(shoulder, target, L1, L2, bend):
    S = np.asarray(shoulder, float)
    T = np.asarray(target, float)
    d = T - S
    dist = np.linalg.norm(d)
    dist_c = np.clip(dist, abs(L1 - L2) + 1, L1 + L2 - 0.5)
    T = S + d / (dist + 1e-9) * dist_c
    a = math.acos(np.clip((L1 ** 2 + dist_c ** 2 - L2 ** 2) / (2 * L1 * dist_c), -1, 1))
    base = math.atan2(d[1], d[0])
    ang = base + bend * a
    E = S + L1 * np.array([math.cos(ang), math.sin(ang)])
    return E, T


# ---------------- руки ----------------

def hand_shapes(kind):
    """Кисть в локальных координатах: запястье в (0,0), пальцы вверх (-y)."""
    palm = ellipse(0, -22, 17, 19, 0, 28)
    caps, lines = [], []
    if kind == "open":
        caps = [capsule((-11, -34), (-15, -62), 5.6, 5.0), capsule((-2, -38), (-2, -68), 5.8, 5.2),
                capsule((7, -36), (10, -63), 5.6, 5.0), capsule((14, -30), (21, -51), 4.8, 4.2),
                capsule((-14, -16), (-31, -34), 6.4, 5.4)]
        lines = [[(-6, -30), (-7, -40)], [(3, -31), (4, -40)], [(11, -28), (13, -36)]]
    elif kind == "point":
        palm = ellipse(0, -21, 17, 18, 0, 28)
        caps = [capsule((-10, -30), (-12, -72), 5.6, 5.0), ellipse(-1, -37, 7, 6), ellipse(8, -35, 6.5, 6),
                ellipse(14, -30, 5.5, 5.5), capsule((-15, -12), (2, -22), 6.4, 5.6)]
        lines = [[(-4, -33), (-3, -40)], [(4, -32), (5, -38)], [(-10, -20), (0, -24)]]
    elif kind == "peace":
        palm = ellipse(0, -21, 17, 18, 0, 28)
        caps = [capsule((-9, -32), (-17, -72), 5.6, 5.0), capsule((3, -34), (11, -72), 5.8, 5.2),
                ellipse(10, -32, 6, 6), ellipse(15, -27, 5, 5), capsule((-15, -12), (2, -22), 6.4, 5.6)]
        lines = [[(-2, -32), (0, -42)], [(-10, -20), (0, -24)]]
    elif kind == "fist":
        palm = ellipse(0, -21, 17, 18, 0, 28)
        caps = [ellipse(-11, -36, 6.5, 6), ellipse(-2, -38, 7, 6.2), ellipse(7, -36, 6.5, 6),
                ellipse(14, -31, 5.5, 5.5), capsule((-15, -12), (0, -24), 6.4, 5.6)]
        lines = [[(-6, -33), (-6, -40)], [(3, -34), (3, -41)], [(11, -31), (11, -37)]]
    return [palm] + caps, lines


def draw_hand(cv, T, wrist, angle, kind, side, scale=1.0, key=0, w=W_OUT):
    shapes, lines = hand_shapes(kind)
    c, s = math.cos(angle), math.sin(angle)

    def X(P):
        P = np.asarray(P, float).copy() * scale
        if side > 0:
            P[:, 0] *= -1
        Q = np.stack([c * P[:, 0] - s * P[:, 1] + wrist[0], s * P[:, 0] + c * P[:, 1] + wrist[1]], 1)
        return T(Q)
    polys = [X(p) for p in shapes]
    sc = T.scale() / BODY_S
    cv.union(polys, C["skin"], w * sc * 0.85, key=key)
    for k, L in enumerate(lines):
        cv.stroke(X(np.array(L, float)), 1.6 * sc, key=key + 10 + k, taper=(0.3, 0.3))


def draw_arm(cv, T, side, arm, key):
    """arm = dict(target=(x,y) в координатах тела | None, hand='open', hang=deg, bend=±1, hscale, hrot)."""
    S = np.array([side * 128.0, 222.0])
    L1, L2 = 118.0, 112.0
    tgt = arm.get("target")
    if tgt is None:
        hang = math.radians(arm.get("hang", 6))
        E = S + L1 * np.array([side * math.sin(hang), math.cos(hang)])
        Wp = E + L2 * np.array([side * math.sin(hang * 0.6), math.cos(hang * 0.6)])
    else:
        E, Wp = ik(S, tgt, L1, L2, arm.get("bend", side))
    sc = T.scale() / BODY_S
    up = capsule(S, E, 25, 21)
    fo = capsule(E, Wp + (Wp - E) / np.linalg.norm(Wp - E) * -6, 21, 19)
    cv.union([T(up), T(fo)], C["blazer"], W_OUT * sc, key=key)
    # кант на рукаве и складка на локте
    d = (Wp - E) / np.linalg.norm(Wp - E)
    n = np.array([-d[1], d[0]])
    cuff_c = Wp - d * 10
    cv.stroke(T(np.array([cuff_c + n * 19, cuff_c - n * 19])), 2.6 * sc, color=C["pipe"], key=key + 1)
    cv.stroke(T(np.array([E - n * 8 - d * 4, E + n * 2, E + n * 10 + d * 6])), 1.6 * sc, key=key + 2)
    # манжета рубашки
    cuff = capsule(Wp - d * 4, Wp + d * 2, 15, 15, 8)
    cv.union([T(cuff)], C["shirt"], W_OUT * sc * 0.8, key=key + 3)
    hand = arm.get("hand", "open")
    ang = math.atan2(d[1], d[0]) + math.pi / 2 + math.radians(arm.get("hrot", 0)) * (1 if side < 0 else -1) * -1
    draw_hand(cv, T, Wp + d * 2, ang, hand, side, arm.get("hscale", 1.0), key=key + 20)


# ---------------- лицо ----------------

def draw_eye(cv, T, side, state, key, look=(0, 0), brow="neutral"):
    sg = side
    cx = 44 * sg
    iris_c, iris_d = (C["irisL"], C["irisL_d"]) if side < 0 else (C["irisR"], C["irisR_d"])
    M = (lambda P: np.asarray(P, float)) if side < 0 else mir
    sc = T.scale()
    if state in ("happy", "wink"):
        cv.stroke(T(M(catmull([(-72, 52), (-60, 34), (-44, 28), (-28, 34), (-16, 50)]))), 4.6 * sc, key=key, taper=(0.25, 0.3))
        cv.stroke(T(M(catmull([(-76, 40), (-82, 34)]))), 2.2 * sc, key=key + 1)
        return
    if state == "closed":
        cv.stroke(T(M(catmull([(-74, 46), (-58, 56), (-42, 58), (-26, 54), (-16, 46)]))), 4.4 * sc, key=key)
        return
    lid = {"open": 0, "half": 16, "angry": 6, "shock": -4, "spiral": 0, "sad": 6}.get(state, 0)
    white = M(catmull([(-76, 32 + lid), (-74, 56), (-58, 72), (-40, 74), (-22, 66), (-14, 44), (-16, 26 + lid * 0.8),
                       (-30, 14 + lid), (-48, 10 + lid), (-66, 16 + lid)], closed=True))
    if state == "angry":
        white = M(catmull([(-76, 26), (-74, 56), (-58, 72), (-40, 74), (-22, 66), (-14, 44), (-16, 34),
                           (-30, 26), (-48, 18), (-66, 16)], closed=True))
    cv.fill(T(white), C["eye_white"])
    lx, ly = look
    if state == "shock":
        ir = ellipse(-44 + lx, 44 + ly, 10, 11, 0, 24)
        cv.shape(T(M(ir)), iris_c, 2.2 * sc, clip=[T(white)], key=key + 2)
        cv.fill(T(M(ellipse(-44 + lx, 44 + ly, 4, 4.5))), LINE)
    else:
        ir = ellipse(-44 + lx, 44 + ly, 21, 23, 0, 36)
        cv.fill(T(M(ir)), iris_c, clip=[T(white)])
        cv.fill(T(M(ellipse(-44 + lx, 36 + ly, 21, 13))), iris_d, alpha=0.65, clip=[T(white), T(M(ir))])
        if state == "spiral":
            sp = np.array([[-44 + lx + r * math.cos(a), 44 + ly + r * math.sin(a)]
                           for a, r in zip(np.linspace(0, 4 * math.pi, 60), np.linspace(1, 18, 60))])
            cv.stroke(T(M(sp)), 2.0 * sc, key=key + 5)
        else:
            cv.fill(T(M(ellipse(-44 + lx, 45 + ly, 9.5, 10.5))), iris_d, clip=[T(white)])
            # крестик-блик в зрачке
            crs = np.array([(-2, -7), (2, -7), (2, -2), (7, -2), (7, 2), (2, 2), (2, 7), (-2, 7), (-2, 2), (-7, 2), (-7, -2), (-2, -2)], float)
            cv.fill(T(M(crs + [-44 + lx, 45 + ly])), C["gold"], clip=[T(white)])
            cv.fill(T(M(ellipse(-53 + lx, 34 + ly, 5, 6, 0.4, 16))), C["white"], clip=[T(white)])
    # веко
    top = M(catmull([(-14, 28 + lid * 0.8), (-28, 14 + lid), (-48, 9 + lid), (-66, 15 + lid), (-80, 30 + lid * 0.6)]))
    if state == "angry":
        top = M(catmull([(-14, 36), (-30, 25), (-48, 17), (-66, 15), (-80, 24)]))
    cv.stroke(T(top), 6.0 * sc, key=key + 4, taper=(0.35, 0.08))
    cv.stroke(T(M(catmull([(-78, 26 + lid * 0.6), (-86, 20 + lid * 0.6)]))), 2.6 * sc, key=key + 6, taper=(0.1, 0.7))
    cv.stroke(T(M(catmull([(-66, 68), (-50, 75), (-32, 72)]))), 1.8 * sc, key=key + 7, taper=(0.4, 0.4))
    cv.stroke(T(M(catmull([(-62, 84), (-44, 88), (-28, 84)]))), 1.4 * sc, color=C["skin_sh"] * 0.85, key=key + 8, taper=(0.4, 0.4))


def draw_brow(cv, T, side, brow, key):
    M = (lambda P: np.asarray(P, float)) if side < 0 else mir
    pts = {"neutral": [(-68, -12), (-48, -18), (-28, -15)], "angry": [(-70, -20), (-48, -14), (-26, -2)],
           "sad": [(-70, -6), (-48, -14), (-28, -24)], "smug": [(-70, -16), (-48, -22), (-26, -14)],
           "up": [(-70, -24), (-48, -30), (-28, -26)]}[brow]
    cv.stroke(T(M(catmull(pts))), 3.6 * T.scale(), key=key, taper=(0.15, 0.5))


def draw_mouth(cv, T, m, key):
    kind = m.get("kind", "smirk")
    v, op = m.get("vowel"), m.get("open", 0)
    sc = T.scale()
    y = 106
    if kind == "smirk" or (kind == "sing" and (v is None or op < 0.12 or v == "m")):
        cv.stroke(T(catmull([(-14, y - 2), (-5, y + 2), (6, y + 2), (15, y - 3)])), 2.4 * sc, key=key, taper=(0.3, 0.3))
        cv.shape(T(np.array([(5, y + 1.5), (10, y + 1), (8, y + 7)])), C["white"], 1.4 * sc, key=key + 1)
        return
    if kind == "flat":
        cv.stroke(T(np.array([(-12, y), (12, y + 1)])), 2.4 * sc, key=key)
        return
    if kind == "wavy":
        cv.stroke(T(catmull([(-16, y + 2), (-8, y - 2), (0, y + 2), (8, y - 2), (16, y + 2)])), 2.2 * sc, key=key)
        return
    if kind == "tongue":
        cv.stroke(T(catmull([(-16, y - 4), (-6, y + 2), (6, y + 2), (16, y - 4)])), 2.4 * sc, key=key)
        tg = catmull([(-1, y + 1), (-1, y + 12), (6, y + 18), (13, y + 12), (12, y + 1)])
        cv.shape(T(tg), C["tongue"], 2.0 * sc, key=key + 1)
        cv.stroke(T(np.array([(6, y + 4), (6, y + 11)])), 1.3 * sc, key=key + 2)
        return
    if kind == "grin":
        wd, hg = 26, 20 * (0.4 + 0.6 * max(op, 0.5))
    else:
        wd = {"a": 15, "o": 10, "u": 7, "i": 18, "e": 15}.get(v, 13)
        hg = {"a": 20, "o": 18, "u": 11, "i": 9, "e": 14}.get(v, 12) * (0.35 + 0.65 * op)
    top = y - 3
    Mo = catmull([(-wd, top), (-wd * 0.4, top + 2), (wd * 0.4, top + 2), (wd, top), (wd * 0.7, top + hg * 0.8),
                  (0, top + hg), (-wd * 0.7, top + hg * 0.8)], closed=True)
    cv.fill(T(Mo), C["mouth"])
    if hg > 9:
        cv.fill(T(ellipse(0, top + hg * 0.85, wd * 0.6, hg * 0.3)), C["tongue"], clip=[T(Mo)])
    if kind == "grin":
        teeth = [(-wd, top)]
        n = 7
        for k in range(n):
            x0 = -wd + 2 * wd * k / n
            teeth += [(x0 + wd / n, top + 7), (x0 + 2 * wd / n, top + 1)]
        teeth += [(wd, top - 2), (-wd, top - 2)]
        cv.fill(T(np.array(teeth)), C["white"], clip=[T(Mo)])
        bot = [(-wd * 0.7, top + hg * 0.8)]
        for k in range(5):
            x0 = -wd * 0.7 + 1.4 * wd * k / 5
            bot += [(x0 + 0.7 * wd / 5, top + hg * 0.8 - 6), (x0 + 1.4 * wd / 5, top + hg * 0.85)]
        bot += [(wd * 0.7, top + hg + 4), (-wd * 0.7, top + hg + 4)]
        cv.fill(T(np.array(bot)), C["white"], clip=[T(Mo)])
    else:
        cv.fill(T(np.array([(4, top), (11, top), (8, top + 7)])), C["white"], clip=[T(Mo)])
    cv.stroke(T(Mo), 2.3 * sc, closed=True, key=key + 3)


# ---------------- персонаж целиком ----------------

TIPS = [(-104, -36), (-96, 18), (-84, -20), (-71, 6), (-58, -30), (-45, -2), (-31, -34), (-15, 32),
        (-4, -28), (9, 8), (22, -34), (37, -4), (51, -30), (66, 6), (80, -22), (95, 20), (106, -36)]


def draw(cv, pose, root):
    """root: Xf из координат тела в кадр. pose — словарь (см. scenes.py)."""
    sc = root.scale()
    hr = math.radians(pose.get("head_rot", 0))
    head = Xf(tx=pose.get("hx", 0), ty=pose.get("hy", 0), rot=hr, pivot=NECK, parent=root)
    hs = head.scale()
    k = 0
    sway = pose.get("sway", 0.0)

    def bend(P, amount, y0=40):
        P = np.asarray(P, float).copy()
        f = np.clip((P[:, 1] - y0) / 90, 0, 1) ** 1.4
        P[:, 0] += f * amount
        return P

    # задние волосы
    back = [(-108, 116), (-124, 104), (-120, 96), (-132, 70), (-126, 40), (-124, -50), (-104, -122), (-60, -162), (0, -174), (60, -162), (104, -122),
            (124, -50), (126, 40), (134, 72), (122, 96), (128, 110), (110, 114), (96, 98), (88, 124), (70, 104), (0, 110), (-70, 104), (-88, 124), (-98, 98)]
    bp = bend(catmull(back, closed=True), sway)
    cv.shape(head(bp), C["hair"], W_OUT * hs, key=1)
    inner = bend(catmull([(-126, 40), (-112, 112), (-98, 98), (-88, 124), (-70, 104), (0, 110), (70, 104), (88, 124),
                          (96, 98), (114, 112), (126, 40), (0, 30)], closed=True), sway)
    cv.fill(head(inner), C["hair_in"], clip=[head(bp)])
    cv.fill(head(bend(catmull([(-126, 60), (-100, 80), (0, 70), (100, 80), (126, 60), (126, 130), (-126, 130)], closed=True), sway)),
            C["hair_in_sh"], clip=[head(bp)])
    cv.stroke(head(bp), W_OUT * hs, closed=True, key=2)
    for side in (-1, 1):
        for j, x in enumerate((112, 120)):
            cv.stroke(head(bend(catmull([(side * (x - 6), 20), (side * x, 60), (side * (x + 2), 96)]), sway)), 1.4 * hs, key=3 + j + side * 7, taper=(0.4, 0.4))

    body = Xf(sx=BODY_S, sy=BODY_S, pivot=BODY_PIVOT, parent=root)
    # шея
    neck = Xf(rot=hr * 0.3, pivot=(0, 200), parent=root)
    nk = catmull([(-22, 100), (-22, 150), (-24, 178), (0, 188), (24, 178), (22, 150), (22, 100)], closed=True)
    cv.shape(neck(nk), C["skin"], W_OUT * sc * 0.9, key=3)
    cv.fill(neck(catmull([(-25, 100), (-26, 150), (0, 164), (26, 150), (25, 100)], closed=True)), C["skin_sh"], clip=[neck(nk)])

    # тело: рубашка, галстук, пиджак
    shirt = catmull([(-40, 180), (40, 180), (36, 300), (20, 720), (-20, 720), (-36, 300)], closed=True)
    cv.shape(body(shirt), C["shirt"], W_OUT * sc, key=4)
    cv.fill(body(catmull([(-40, 180), (40, 180), (30, 214), (0, 226), (-30, 214)], closed=True)), C["shirt_sh"], clip=[body(shirt)])
    tie_sw = pose.get("tie", 0.0)
    tie = catmull([(-9, 206), (-15 + tie_sw, 290), (-2 + tie_sw * 1.6, 330), (13 + tie_sw, 292), (9, 206)], closed=True)
    cv.shape(body(tie), C["tie"], W_OUT * sc * 0.85, key=5)
    cv.fill(body(catmull([(1, 210), (13 + tie_sw, 292), (-2 + tie_sw * 1.6, 330), (4 + tie_sw, 290)], closed=True)), C["tie_sh"], clip=[body(tie)])
    knot = catmull([(-12, 190), (12, 190), (9, 210), (-9, 210)], closed=True)
    cv.shape(body(knot), C["tie"], W_OUT * sc * 0.85, key=6)
    for side in (-1, 1):
        M = (lambda P: np.asarray(P, float)) if side < 0 else mir
        coll = M(catmull([(-30, 178), (-36, 196), (-14, 220), (-4, 200)], closed=True))
        cv.shape(body(coll), C["shirt"], W_OUT * sc * 0.8, key=7 + side)
    blz = [(-30, 176), (-74, 192), (-128, 206), (-162, 236), (-176, 300), (-186, 720), (-8, 720), (-24, 330),
           (-48, 262), (-40, 240), (-60, 226), (-36, 190)]
    for side in (-1, 1):
        M = (lambda P: np.asarray(P, float)) if side < 0 else mir
        P = M(catmull(blz, closed=True))
        cv.shape(body(P), C["blazer"], W_OUT * sc, key=10 + side)
        cv.fill(body(M(catmull([(-176, 300), (-150, 280), (-120, 330), (-110, 720), (-186, 720)], closed=True))),
                C["blazer_sh"], clip=[body(P)])
        cv.stroke(body(M(catmull([(-36, 192), (-60, 226), (-40, 240), (-48, 262), (-24, 330), (-8, 720)]))),
                  3.0 * sc, color=C["pipe"], key=12 + side)
        cv.stroke(body(M(catmull([(-74, 194), (-126, 210), (-150, 226)]))), 2.6 * sc, color=C["pipe"], key=14 + side)
        cv.stroke(body(M(catmull([(-60, 226), (-40, 240)]))), 2.0 * sc, key=16 + side)
        for by in (370, 450, 530):
            cv.shape(body(M(ellipse(-38, by, 6, 6, 0, 14))), C["gold"], 1.6 * sc, key=18 + side + by)
        cv.stroke(body(M(catmull([(-110, 300), (-104, 340), (-108, 380)]))), 1.6 * sc, key=20 + side, taper=(0.4, 0.4))

    # уши
    for side in (-1, 1):
        ear = ellipse(side * 96, 42, 12, 18, side * 0.2, 20)
        cv.shape(head(ear), C["skin"], W_OUT * hs * 0.8, key=22 + side)
        cv.stroke(head(catmull([(side * 94, 32), (side * 100, 42), (side * 95, 52)])), 1.4 * hs, key=24 + side)

    # лицо
    jaw = [(-94, -30), (-95, 10), (-88, 52), (-70, 92), (-42, 122), (-12, 136), (0, 138), (12, 136), (42, 122),
           (70, 92), (88, 52), (95, 10), (94, -30)]
    face = catmull(jaw + [(60, -110), (0, -120), (-60, -110)], closed=True)
    cv.fill(head(face), C["skin"])
    sh = [(-96, -40)] + [(x, y + 12) for x, y in TIPS[1:-1]] + [(96, -40)]
    cv.fill(head(catmull(sh, closed=True)), C["skin_sh"], clip=[head(face)], alpha=0.85)
    cv.stroke(head(catmull(jaw[1:-1])), W_OUT * hs, key=26, taper=(0.06, 0.06))
    if pose.get("blush", 0.5) > 0:
        b = pose.get("blush", 0.5)
        for side in (-1, 1):
            cv.fill(head(ellipse(side * 56, 86, 20, 9)), C["blush"], alpha=0.45 * b)
            for j in range(3):
                x = side * (46 + j * 9)
                cv.stroke(head(np.array([(x - 3, 92), (x + 3, 82)])), 1.6 * hs, color=hexc("#e2486d"), key=27 + j + side * 5, overshoot=0)
    cv.stroke(head(np.array([(1, 80), (-1, 84)])), 2.0 * hs, key=30, overshoot=0)
    for side in (-1, 1):
        st = pose.get("eyeL" if side < 0 else "eyeR", pose.get("eyes", "open"))
        draw_eye(cv, head, side, st, key=40 + side * 20, look=pose.get("look", (0, 0)))
    draw_mouth(cv, head, pose.get("mouth", {"kind": "smirk"}), key=80)
    for j, (x, y) in enumerate(pose.get("sweat", [])):
        drop = catmull([(x, y - 12), (x + 6, y + 2), (x, y + 7), (x - 6, y + 2)], closed=True)
        cv.shape(head(drop), hexc("#bfe8ff"), 2.0 * hs, key=90 + j)

    # передние пряди по бокам
    for side in (-1, 1):
        M = (lambda P: np.asarray(P, float)) if side < 0 else mir
        lock = M(path([(-86, -40), (-84, 30), (-84, 80), (-80, 132)], [(-80, 132), (-90, 112), (-98, 136)],
                      [(-98, 136), (-104, 90), (-110, 20), (-108, -50)]))
        lock = bend(lock, sway * 0.7, 30)
        cv.shape(head(lock), C["hair"], W_OUT * hs, key=100 + side)
        cv.stroke(head(M(bend(catmull([(-96, 0), (-94, 60), (-92, 110)]), sway * 0.7, 30))), 1.4 * hs, key=102 + side, taper=(0.5, 0.4))
    # чёлка
    pieces = []
    for a, b in zip(TIPS, TIPS[1:]):
        m = ((a[0] + b[0]) / 2 + (2 if b[1] > a[1] else -2), (a[1] + b[1]) / 2 + (-5 if b[1] > a[1] else 5))
        pieces.append([a, m, b])
    edge = path(*pieces)
    top = catmull([(106, -36), (112, -84), (98, -128), (58, -160), (0, -172), (-58, -160), (-98, -128), (-112, -84), (-104, -36)])
    bang = np.vstack([edge, top[1:]])
    cv.fill(head(bang), C["hair"])
    for side in (-1, 1):
        cv.fill(head(np.array([(side * 112, -84), (side * 100, -36), (side * 80, -50), (side * 92, -100)])), C["hair_sh"], clip=[head(bang)], alpha=0.7)
    hl = catmull([(-80, -112), (-50, -128), (-20, -120), (10, -130), (40, -122), (70, -112)])
    cv.stroke(head(hl), 5.0 * hs, color=C["white"], key=110, taper=(0.3, 0.3))
    cv.stroke(head(edge), W_OUT * hs, key=111, taper=(0.04, 0.04))
    cv.stroke(head(top), W_OUT * hs, key=112, taper=(0.04, 0.04))
    for j, (a, b) in enumerate(zip(TIPS[1::2], TIPS[2::2])):
        cv.stroke(head(catmull([(b[0] + 2, b[1] - 6), (b[0] + 5, b[1] - 30), (b[0] + 3, b[1] - 56)])), 1.4 * hs, key=113 + j, taper=(0.5, 0.2))
    # ахоге
    asw = 5 * math.sin(pose.get("aho", 0))
    aho = path([(-6, -168), (-2 + asw * 0.3, -192), (14 + asw, -210), (32 + asw, -206)],
               [(32 + asw, -206), (16 + asw, -200), (6 + asw * 0.5, -188), (6, -169)])
    cv.shape(head(aho), C["hair"], W_OUT * hs * 0.9, key=120)
    # брови поверх чёлки (приём аниме)
    for side in (-1, 1):
        draw_brow(cv, head, side, pose.get("brow", "neutral"), key=125 + side)
    # заколки: красный крестик и синяя
    for j, a in enumerate((0.75, -0.75)):
        bar = capsule((-66 - 15 * math.cos(a), -70 - 15 * math.sin(a)), (-66 + 15 * math.cos(a), -70 + 15 * math.sin(a)), 4.6, 4.6, 8)
        cv.shape(head(bar), C["clip"], 2.4 * hs, key=130 + j)
    cl = capsule((60, -58), (84, -74), 4.2, 4.2, 8)
    cv.shape(head(cl), C["clip2"], 2.4 * hs, key=133)

    # руки (перед телом)
    for side, key in ((-1, "armL"), (1, "armR")):
        arm = pose.get(key)
        if arm is not None:
            a = dict(arm)
            if a.get("target") is not None:
                a["target"] = to_body(a["target"])
            draw_arm(cv, body, side, a, key=200 + side * 50)
    return head
