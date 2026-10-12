"""Кисти, нарисованные вручную (по точкам) под конкретные ракурсы — без универсального рига.

Локальные координаты кисти: запястье в (0, 0), пальцы вверх (-y), длина кисти ~105.
Правая кисть персонажа. Левая получается зеркалом по x.
"""
import math
import numpy as np
from flat import catmull, path, ellipse, LINE

SKIN = np.array([1.0, 0.902, 0.863])
SKIN_SH = np.array([0.95, 0.706, 0.722])


def open_hand(view="back", spread=0.0):
    """Раскрытая кисть, пальцы вместе (spread>0 — веером). view: back | palm."""
    s = spread
    # (основание-лево, кончик, основание-право, ширина кончика)
    fingers = [  # указательный, средний, безымянный, мизинец
        ((-22, -56), (-25 - 6 * s, -97), (-11, -60)),
        ((-10, -60), (-3, -106), (4, -62)),
        ((4, -61), (11 + 5 * s, -100), (16, -58)),
        ((16, -56), (25 + 9 * s, -84), (27, -52)),
    ]
    pieces = [[(19, 0), (25, -24), (27, -52)]]
    pts_fill = []
    # мизинец → указательный (по часовой, если смотреть с тыльной стороны правой кисти)
    for (bl, tip, br) in reversed(fingers):
        bl, tip, br = np.array(bl, float), np.array(tip, float), np.array(br, float)
        d = tip - (bl + br) / 2
        L = np.linalg.norm(d)
        u = d / L
        n = np.array([-u[1], u[0]])
        half = np.linalg.norm(br - bl) / 2 * 0.95
        th = half * 0.8                      # к кончику палец сужается
        c = tip - u * th
        right_side = [br, br + d * 0.5 + n * (half - th) * -0.2 + n * 0.8, c + n * th]
        cap = [c + n * th, c + u * th * 0.72 + n * th * 0.72, tip, c + u * th * 0.72 - n * th * 0.72, c - n * th]
        left_side = [c - n * th, bl + d * 0.5 - n * 0.3, bl]
        pieces += [right_side, cap, left_side]
    # большой палец
    pieces += [[(-22, -56), (-23, -52), (-24, -47)],
               [(-24, -47), (-32, -50), (-40, -58)],
               [(-40, -58), (-44, -65), (-50, -67), (-55, -61), (-54, -55)],
               [(-54, -55), (-48, -42), (-38, -27), (-27, -12), (-21, 0)]]
    outline = path(*pieces)
    lines = []
    # линии между пальцами
    for (bl, tip, br) in fingers[:-1]:
        lines.append(np.array([br, np.array(br) + np.array([0.5, 9])]))
    if view == "back":
        # ногти и костяшки
        for (bl, tip, br) in fingers:
            tip = np.array(tip, float)
            w = abs(br[0] - bl[0]) * 0.32
            lines.append(catmull([(tip[0] - w, tip[1] + 11), (tip[0], tip[1] + 6), (tip[0] + w, tip[1] + 11)]))
            m = (np.array(bl) + np.array(br)) / 2
            lines.append(catmull([m + (-3, 4), m + (0, 2), m + (3, 4)]))
        lines.append(catmull([(-50, -60), (-46, -60), (-44, -56)]))
        lines.append(catmull([(-30, -36), (-26, -30), (-24, -22)]))
        shade = None
    else:
        lines += [catmull([(-26, -20), (-10, -36), (8, -44)]), catmull([(-24, -40), (0, -46), (20, -42)]),
                  catmull([(-18, -10), (-24, -26), (-26, -40)])]
        shade = catmull([(-21, 0), (-24, -14), (-30, -30), (-20, -26), (-8, -16), (19, 0)], closed=True)
    return outline, lines, shade


def point_front():
    """Указательный палец направлен на зрителя (ракурс 3/4): кончик ближе, поэтому крупнее."""
    fist = catmull([(-28, -4), (-34, -28), (-30, -50), (-14, -58), (8, -60), (28, -54), (36, -36), (32, -10),
                    (20, 4), (-12, 6)], closed=True)
    finger = path([(-28, -46), (-34, -62), (-40, -80)],
                  [(-40, -80), (-42, -92), (-34, -102), (-20, -104), (-10, -96), (-8, -86)],
                  [(-8, -86), (-6, -70), (-6, -54)])
    nail = catmull([(-34, -94), (-28, -100), (-18, -100), (-14, -94), (-22, -90)], closed=True)
    joints = [catmull([(-38, -72), (-24, -68), (-10, -73)])]
    curls = [catmull([(-4, -56), (-2, -40), (-4, -24)]), catmull([(12, -57), (14, -40), (12, -24)]),
             catmull([(26, -50), (28, -36), (26, -22)]),
             catmull([(-6, -50), (3, -54), (11, -50)]), catmull([(12, -51), (19, -54), (26, -48)])]
    thumb = path([(-32, -22), (-18, -30), (2, -32), (12, -30)], [(12, -30), (17, -25), (13, -19)],
                 [(13, -19), (-4, -16), (-22, -10), (-30, -8)])
    return fist, finger, nail, joints + curls, thumb


def fist_front():
    """Кулак костяшками к зрителю."""
    fist = catmull([(-30, -4), (-34, -30), (-28, -56), (-10, -64), (12, -64), (30, -56), (36, -32), (32, -6), (18, 6), (-16, 6)], closed=True)
    lines = [catmull([(-24, -44), (-22, -30), (-24, -16)]), catmull([(-8, -48), (-6, -32), (-8, -18)]),
             catmull([(8, -48), (9, -32), (8, -18)]), catmull([(22, -44), (23, -30), (22, -18)])]
    knuck = [catmull([(-26, -52), (-17, -58), (-9, -54)]), catmull([(-8, -56), (0, -60), (8, -56)]),
             catmull([(9, -55), (17, -58), (25, -52)])]
    thumb = path([(-33, -22), (-18, -28), (4, -28), (14, -24)], [(14, -24), (17, -18), (12, -14)],
                 [(12, -14), (-6, -12), (-24, -8), (-31, -8)])
    lines = [catmull([(-24, -54), (-22, -42), (-24, -30)]), catmull([(-8, -58), (-6, -44), (-8, -30)]),
             catmull([(8, -58), (9, -44), (8, -30)]), catmull([(22, -54), (23, -40), (22, -28)])]
    return fist, lines + knuck, thumb


def place(P, wrist, angle, scale=1.0, mirror=False):
    P = np.asarray(P, float).copy() * scale
    if mirror:
        P[:, 0] *= -1
    c, s = math.cos(angle), math.sin(angle)
    return np.stack([c * P[:, 0] - s * P[:, 1] + wrist[0], s * P[:, 0] + c * P[:, 1] + wrist[1]], 1)


def draw_open(cv, T, wrist, angle, view="back", mirror=False, scale=1.0, spread=0.0, key=0, w=3.0):
    o, lines, shade = open_hand(view, spread)
    X = lambda P: T(place(P, wrist, angle, scale, mirror))
    sc = T.scale()
    cv.fill(X(o), SKIN)
    if shade is not None:
        cv.fill(X(shade), SKIN_SH, clip=[X(o)], alpha=0.7)
    cv.stroke(X(o), w * sc, closed=True, key=key)
    for k, L in enumerate(lines):
        cv.stroke(X(L), 1.5 * sc, key=key + 1 + k, taper=(0.3, 0.3))


def draw_point_front(cv, T, wrist, angle, mirror=False, scale=1.0, key=0, w=3.0):
    fist, finger, nail, lines, thumb = point_front()
    X = lambda P: T(place(P, wrist, angle, scale, mirror))
    sc = T.scale()
    cv.fill(X(fist), SKIN)
    cv.fill(X(catmull([(-30, -6), (34, -10), (22, 4), (-14, 6)], closed=True)), SKIN_SH, clip=[X(fist)], alpha=0.8)
    cv.stroke(X(fist), w * sc, closed=True, key=key)
    for k, L in enumerate(lines[2:]):
        cv.stroke(X(L), 1.6 * sc, key=key + 10 + k, taper=(0.2, 0.4))
    cv.fill(X(thumb), SKIN)
    cv.stroke(X(thumb), w * 0.8 * sc, closed=True, key=key + 2)
    cv.fill(X(finger), SKIN)
    cv.stroke(X(finger), w * sc, closed=True, key=key + 3)
    for k, L in enumerate(lines[:2]):
        cv.stroke(X(L), 1.4 * sc, key=key + 4 + k, taper=(0.3, 0.3))
    cv.fill(X(nail), np.array([1.0, 0.95, 0.95]))
    cv.stroke(X(nail), 1.5 * sc, closed=True, key=key + 6)


def draw_fist_front(cv, T, wrist, angle, mirror=False, scale=1.0, key=0, w=3.0):
    fist, lines, thumb = fist_front()
    X = lambda P: T(place(P, wrist, angle, scale, mirror))
    sc = T.scale()
    cv.fill(X(fist), SKIN)
    cv.stroke(X(fist), w * sc, closed=True, key=key)
    for k, L in enumerate(lines):
        cv.stroke(X(L), 1.5 * sc, key=key + 1 + k, taper=(0.3, 0.3))
    cv.fill(X(thumb), SKIN)
    cv.stroke(X(thumb), w * 0.85 * sc, closed=True, key=key + 20)
