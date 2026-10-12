"""Позы, нарисованные под конкретные моменты (ключевые рисунки), и промежуточные фазы.

Координаты — как у головы персонажа (0,0 между глаз), без отдельного масштаба тела.
Руки собираются из сегментов-рукавов, у каждого своя осевая линия и ширина
(ширина задаёт перспективу: ближе к зрителю — шире).
"""
import math
import numpy as np
from flat import catmull, ellipse, offset_curve, LINE, hexc
from chara import C, mir
import hands

W_OUT = 3.4


# ---------------- торс ----------------

def torso(cv, root, pose):
    sc = root.scale()
    shirt = catmull([(-34, 176), (34, 176), (28, 262), (0, 470), (-28, 262)], closed=True)
    cv.shape(root(shirt), C["shirt"], W_OUT * sc, key=400)
    cv.fill(root(catmull([(-34, 176), (34, 176), (26, 200), (0, 212), (-26, 200)], closed=True)), C["shirt_sh"], clip=[root(shirt)])
    sw = pose.get("tie", 0.0)
    tie = catmull([(-8, 202), (-13 + sw, 282), (0 + sw * 1.5, 312), (13 + sw, 282), (8, 202)], closed=True)
    cv.shape(root(tie), C["tie"], W_OUT * sc * 0.85, key=401)
    cv.fill(root(catmull([(1, 206), (13 + sw, 282), (0 + sw * 1.5, 312), (4 + sw, 282)], closed=True)), C["tie_sh"], clip=[root(tie)])
    knot = catmull([(-11, 186), (11, 186), (8, 205), (-8, 205)], closed=True)
    cv.shape(root(knot), C["tie"], W_OUT * sc * 0.85, key=402)
    for side in (-1, 1):
        M = (lambda P: np.asarray(P, float)) if side < 0 else mir
        cv.shape(root(M(catmull([(-28, 174), (-34, 192), (-13, 214), (-4, 195)], closed=True))), C["shirt"], W_OUT * sc * 0.8, key=403 + side)
    blz = [(-28, 172), (-66, 186), (-112, 198), (-140, 210), (-152, 226), (-150, 262), (-144, 340), (-140, 480),
           (-8, 480), (-20, 322), (-42, 258), (-34, 238), (-52, 222), (-32, 186)]
    for side in (-1, 1):
        M = (lambda P: np.asarray(P, float)) if side < 0 else mir
        P = M(catmull(blz, closed=True))
        cv.shape(root(P), C["blazer"], W_OUT * sc, key=405 + side)
        cv.fill(root(M(catmull([(-150, 262), (-118, 280), (-104, 480), (-150, 480)], closed=True))), C["blazer_sh"], clip=[root(P)])
        cv.stroke(root(M(catmull([(-32, 188), (-52, 222), (-34, 238), (-42, 258), (-20, 322), (-8, 480)]))), 3.0 * sc, color=C["pipe"], key=407 + side)
        cv.stroke(root(M(catmull([(-66, 188), (-112, 200), (-136, 212)]))), 2.6 * sc, color=C["pipe"], key=409 + side)
        cv.stroke(root(M(catmull([(-52, 222), (-34, 238)]))), 2.0 * sc, key=411 + side)
        for by in (360, 430):
            cv.shape(root(M(ellipse(-32, by, 5.5, 5.5, 0, 14))), C["gold"], 1.6 * sc, key=413 + side + by)


# ---------------- рукава ----------------

def tube(P, widths):
    Cc = catmull(P, step=2.5)
    n = len(Cc)
    u = np.linspace(0, 1, n)
    hw = np.interp(u, np.linspace(0, 1, len(widths)), widths) / 2
    left = offset_curve(Cc, hw)
    right = offset_curve(Cc, -hw)
    return Cc, left, right, hw


def draw_segment(cv, root, seg, key, cap_start=True):
    """seg: dict(c=[точки оси], w=[ширины], cuff=0..1 раскрытие манжеты | None, folds=[(u, сторона, длина)])."""
    sc = root.scale()
    Cc, left, right, hw = tube(seg["c"], seg["w"])
    d0 = Cc[1] - Cc[0]
    d0 /= np.linalg.norm(d0) + 1e-9
    a0 = math.atan2(d0[1], d0[0])
    cap = [Cc[0] + hw[0] * np.array([math.cos(a0 + math.pi / 2 + k * math.pi / 10), math.sin(a0 + math.pi / 2 + k * math.pi / 10)])
           for k in range(11)] if cap_start else []
    poly = np.vstack([left, right[::-1]] + ([np.array(cap)] if cap else []))
    cv.fill(root(poly), C["blazer"])
    # тень по нижней стороне рукава
    shade = np.vstack([right, offset_curve(Cc, -hw * 0.25)[::-1]])
    cv.fill(root(shade), C["blazer_sh"], clip=[root(poly)])
    cv.stroke(root(left), W_OUT * sc, key=key, taper=(0.02, 0.02))
    cv.stroke(root(right), W_OUT * sc, key=key + 1, taper=(0.02, 0.02))
    if cap:
        cv.stroke(root(np.array(cap)), W_OUT * sc, key=key + 2, taper=(0.05, 0.05))
    for j, (u, side, L) in enumerate(seg.get("folds", [])):
        i = int(u * (len(Cc) - 1))
        e = left[i] if side > 0 else right[i]
        c = Cc[i]
        dirn = (c - e) / (np.linalg.norm(c - e) + 1e-9)
        t = Cc[min(i + 2, len(Cc) - 1)] - Cc[max(i - 2, 0)]
        t /= np.linalg.norm(t) + 1e-9
        p = [e, e + dirn * L * 0.6 + t * L * 0.25, e + dirn * L - t * L * 0.1]
        cv.stroke(root(catmull(p)), 1.8 * sc, key=key + 10 + j, taper=(0.1, 0.6))
    cuff = seg.get("cuff")
    if cuff is not None:
        end, dirn = Cc[-1], Cc[-1] - Cc[-4]
        dirn /= np.linalg.norm(dirn) + 1e-9
        ang = math.atan2(dirn[1], dirn[0]) + math.pi / 2
        r1, r2 = hw[-1], max(3.0, hw[-1] * cuff)
        # кант у манжеты
        band = [end - dirn * 9 + np.array([math.cos(ang), math.sin(ang)]) * hw[-1] * s for s in np.linspace(-1, 1, 9)]
        band = [b - dirn * (1 - (k / 4 - 1) ** 2) * r2 * 0.6 for k, b in enumerate(band)]
        cv.stroke(root(np.array(band)), 2.6 * sc, color=C["pipe"], key=key + 30)
        op = ellipse(end[0], end[1], r1, r2, ang, 30)
        cv.shape(root(op), C["shirt"], W_OUT * sc * 0.8, key=key + 31)
        if cuff > 0.35:
            cv.fill(root(ellipse(end[0] + dirn[0] * 2, end[1] + dirn[1] * 2, r1 * 0.72, r2 * 0.62, ang, 24)), hexc("#3a2f4a"))
    return Cc[-1], Cc[-1] - Cc[-3]


def draw_arm(cv, root, arm, key):
    """arm: dict(segs=[...], hand=dict(kind, at=(dx,dy) от конца рукава, angle (рад), scale, mirror, view))."""
    end = None
    for k, seg in enumerate(arm["segs"]):
        end, dirn = draw_segment(cv, root, seg, key + k * 50, cap_start=seg.get("cap", True))
    h = arm.get("hand")
    if h:
        dirn = dirn / (np.linalg.norm(dirn) + 1e-9)
        wrist = end + dirn * h.get("inset", -6) + np.array(h.get("at", (0, 0)))
        kind = h["kind"]
        kw = dict(mirror=h.get("mirror", False), scale=h.get("scale", 1.0), key=key + 200, w=W_OUT)
        if kind == "open":
            hands.draw_open(cv, root, wrist, h["angle"], h.get("view", "back"), spread=h.get("spread", 0.0), **kw)
        elif kind == "point":
            hands.draw_point_front(cv, root, wrist, h["angle"], **kw)
        elif kind == "fist":
            hands.draw_fist_front(cv, root, wrist, h["angle"], **kw)


# ---------------- ключевые позы рук ----------------

def arm_down(s):
    return dict(segs=[dict(c=[(s * 136, 228), (s * 158, 300), (s * 166, 390), (s * 168, 480)], w=[40, 46, 44, 44], cap=False,
                           folds=[(0.62, -s, 14)])])


def arm_point(s=1):
    """«アナタ» — палец в зрителя: локоть вниз-вперёд, предплечье к камере (расширяется)."""
    return dict(segs=[
        dict(c=[(s * 136, 228), (s * 170, 280), (s * 176, 330)], w=[40, 46, 46], cap=False, folds=[(0.7, -s, 12)]),
        dict(c=[(s * 176, 334), (s * 160, 290), (s * 132, 236)], w=[46, 54, 62], cuff=0.75,
             folds=[(0.15, s, 16), (0.3, s, 12)]),
    ], hand=dict(kind="point", angle=-0.35 * s, scale=1.45, mirror=s > 0, inset=-10))


def arm_chest(s=-1):
    """«アタシ» — ладонь тыльной стороной на груди."""
    return dict(segs=[
        dict(c=[(s * 136, 228), (s * 166, 290), (s * 166, 346)], w=[40, 46, 44], cap=False, folds=[(0.75, -s, 12)]),
        dict(c=[(s * 166, 350), (s * 120, 330), (s * 64, 300)], w=[44, 42, 42], cuff=0.3, folds=[(0.12, -1, 14)]),
    ], hand=dict(kind="open", view="back", angle=0.62, scale=1.1, mirror=False, at=(4, 4), inset=-4))


def arm_cross(s, front):
    """«キライ» — руки крест-накрест, ладони к зрителю."""
    return dict(segs=[
        dict(c=[(s * 136, 228), (s * 160, 290), (s * 138, 348)], w=[40, 46, 44], cap=False, folds=[(0.8, s, 12)]),
        dict(c=[(s * 140, 352), (s * 60, 312), (-s * 44, 262)], w=[44, 42, 42], cuff=0.35, folds=[(0.1, 1, 12)]),
    ], hand=dict(kind="open", view="palm", angle=-0.42 * s, scale=1.0, mirror=s < 0, inset=-4, spread=0.25))


def arm_pre_cross(s):
    """Замах перед «✕»: локти в стороны, кулаки у плеч."""
    return dict(segs=[
        dict(c=[(s * 136, 228), (s * 182, 280), (s * 196, 326)], w=[40, 48, 46], cap=False, folds=[(0.8, s, 12)]),
        dict(c=[(s * 196, 330), (s * 180, 270), (s * 150, 226)], w=[46, 44, 46], cuff=0.4),
    ], hand=dict(kind="fist", angle=-0.6 * s, scale=1.0, mirror=s > 0, inset=-4))


def lerp_arm(a, b, t):
    """Промежуточная фаза: интерполяция осевых линий, ширин и кисти (как «фазовка» у аниматора)."""
    if len(a["segs"]) != len(b["segs"]):
        a = dict(a, segs=a["segs"] + [dict(a["segs"][-1])] * (len(b["segs"]) - len(a["segs"]))) if len(a["segs"]) < len(b["segs"]) else a
        b = dict(b, segs=b["segs"] + [dict(b["segs"][-1])] * (len(a["segs"]) - len(b["segs"]))) if len(b["segs"]) < len(a["segs"]) else b
    segs = []
    for sa, sb in zip(a["segs"], b["segs"]):
        ca = np.array(catmull(sa["c"]))
        cb = np.array(catmull(sb["c"]))
        from flat import resample_n
        n = 6
        c = resample_n(ca, n) * (1 - t) + resample_n(cb, n) * t
        wa = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(sa["w"])), sa["w"])
        wb = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(sb["w"])), sb["w"])
        seg = dict(c=[tuple(p) for p in c], w=list(wa * (1 - t) + wb * t), folds=(sa if t < 0.5 else sb).get("folds", []),
                   cap=(sa if t < 0.5 else sb).get("cap", True))
        ca_, cb_ = sa.get("cuff"), sb.get("cuff")
        if ca_ is not None or cb_ is not None:
            seg["cuff"] = (ca_ or 0.3) * (1 - t) + (cb_ or 0.3) * t
        segs.append(seg)
    out = dict(segs=segs)
    ha, hb = a.get("hand"), b.get("hand")
    if ha and hb and ha["kind"] == hb["kind"]:
        h = dict(hb)
        for k in ("angle", "scale", "inset"):
            h[k] = ha.get(k, 0) * (1 - t) + hb.get(k, 0) * t
        out["hand"] = h
    else:
        h = (ha if t < 0.5 else hb)
        if h:
            out["hand"] = h
    return out


def arms_fn(left, right, front="R"):
    """left — рука у левого края кадра (правая рука персонажа)."""
    def f(cv, root, pose):
        order = [(left, 600), (right, 800)] if front == "R" else [(right, 800), (left, 600)]
        for arm, key in order:
            if arm is not None:
                draw_arm(cv, root, arm, key)
    return f
