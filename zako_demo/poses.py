"""Позы Рэй — каждая нарисована отдельно под свой момент (как иллюстрации в MV).

Каждая функция рисует в координатах панели 427×720 (или сцены 1280×720 для танца).
m — форма рта (по вокалу), подставляется вместо стандартной.
"""
import math
import numpy as np
import rei  # noqa: F401 (добавляет путь к flat.py)
from flat import Xf, ellipse, catmull, hexc
import rei
from rei import Pen, C, head, torso, sleeve, legs, fist

DESK = hexc("#b9988f")
DESK2 = hexc("#9b7d78")
METAL = hexc("#c9c3d6")
DARK = hexc("#4a4458")


def _mouth(m, default):
    return m if m else default


def p_lying(cv, m=None, ox=0):          # いいこいいこして あげようか
    pen = Pen(cv, Xf(tx=230 + ox, ty=300, sx=1.05, sy=1.05))
    torso(pen, hem=520)
    head(pen, "half", _mouth(m, "o"), tilt=14, look=(4, 0))
    sleeve(pen, [(-330, 300), (-160, 272), (0, 262), (110, 250)], 66, 70, hand="fist", hand_ang=0.15, skip=0)
    sleeve(pen, [(-320, 380), (-120, 350), (80, 340), (190, 330)], 70, 74, hand="fist", hand_ang=-0.1, skip=0)


def p_chin(cv, m=None, ox=0):           # 目が合ったね ドキドキしてるんだ
    pen = Pen(cv, Xf(tx=200 + ox, ty=250, sx=0.95, sy=0.95))
    torso(pen, hem=520)
    head(pen, "half", _mouth(m, "o"), tilt=-6, look=(-3, 0))
    pen.shape([(-260, 370), (300, 370), (300, 410), (-260, 410)], DESK, smooth=False)
    pen.shape([(-260, 410), (300, 410), (300, 430), (-260, 430)], DESK2, smooth=False)
    for x in (-150, 190):
        pen.shape([(x, 430), (x + 16, 430), (x + 16, 900), (x, 900)], METAL, smooth=False)
    sleeve(pen, [(-96, 380), (-86, 260), (-56, 150)], 64, 66, hand="fist", hand_ang=-0.2, skip=0)
    sleeve(pen, [(96, 380), (86, 260), (56, 150)], 64, 66, hand="fist", hand_ang=0.2, skip=0)


def p_knees(cv, m=None, ox=0):          # 何興奮してんの 超キモいんだけど
    pen = Pen(cv, Xf(tx=210 + ox, ty=230, sx=1.05, sy=1.05))
    torso(pen, hem=520)
    head(pen, "half", "o", tilt=8, look=(5, 2))
    for x0 in (-46, 46):                 # голени уходят вниз из-под коленей
        pen.shape([(x0 - 36, 170), (x0 + 36, 170), (x0 + 34, 760), (x0 - 34, 760)], C["sock"], smooth=False)
    knees = [(-104, 240), (-100, 140), (-74, 100), (-36, 96), (-4, 118), (6, 118), (36, 92), (76, 96), (102, 136), (104, 240)]
    pen.shape(knees, C["skin"])
    pen.line([(0, 122), (2, 240)], 0.8)
    sleeve(pen, [(-210, 190), (-90, 136), (20, 126), (90, 112)], 66, 70, hand="fist", hand_ang=-0.5, skip=0)
    sleeve(pen, [(230, 190), (120, 158), (20, 154)], 68, 70, hand="fist", hand_ang=0.3, skip=0)


def p_close_mouth(cv, m=None, ox=0):    # крупно: рукав у рта, прищур
    pen = Pen(cv, Xf(tx=230 + ox, ty=290, sx=2.1, sy=2.1))
    torso(pen, hem=520)
    head(pen, "half", "line", tilt=-8, look=(3, 1))
    sleeve(pen, [(-30, 330), (-24, 220), (-10, 128)], 70, 72, hand="fist", hand_ang=-0.7, skip=0)


def p_fists(cv, m=None, ox=0):          # なにそのくそざこ
    pen = Pen(cv, Xf(tx=213 + ox, ty=230, sx=1.0, sy=1.0))
    torso(pen, hem=520)
    head(pen, "open", _mouth(m, "o"), tilt=-4)
    sleeve(pen, [(-62, 200), (-104, 280), (-72, 320), (-34, 250)], 58, 62, hand="fist", hand_ang=-0.4)
    sleeve(pen, [(62, 200), (104, 280), (72, 320), (34, 250)], 58, 62, hand="fist", hand_ang=0.4)


def p_shrug(cv, m=None, ox=0):          # 最底辺よわよわゴミ
    pen = Pen(cv, Xf(tx=213 + ox, ty=230, sx=0.95, sy=0.95))
    torso(pen, hem=520)
    head(pen, "half", _mouth(m, "grin"), tilt=7)
    sleeve(pen, [(-62, 200), (-104, 290), (-150, 300), (-186, 250)], 58, 62, hand="open", hand_ang=-0.5)
    sleeve(pen, [(62, 200), (104, 290), (150, 300), (186, 250)], 58, 62, hand="open", hand_ang=0.5)


def p_cheeks(cv, m=None, ox=0, eyes="xx", zoom=1.0, mouth="w"):   # けどほんとはすきすき
    pen = Pen(cv, Xf(tx=213 + ox, ty=250 + 40 * (zoom - 1), sx=1.0 * zoom, sy=1.0 * zoom))
    torso(pen, hem=520)
    head(pen, eyes, _mouth(m, mouth), tilt=-5)
    sleeve(pen, [(-110, 330), (-104, 220), (-80, 104)], 62, 64, hand="fist", hand_ang=-0.9, skip=0)
    sleeve(pen, [(110, 330), (104, 220), (80, 104)], 62, 64, hand="fist", hand_ang=0.9, skip=0)


def p_hand_mouth(cv, m=None, ox=0):     # なにちょっと
    pen = Pen(cv, Xf(tx=213 + ox, ty=230, sx=1.0, sy=1.0))
    torso(pen, hem=520)
    sleeve(pen, [(-62, 200), (-90, 320), (-92, 440)], 58, 62)
    head(pen, "open", _mouth(m, "open"), tilt=6, look=(-4, 0))
    sleeve(pen, [(80, 340), (64, 230), (30, 118)], 60, 62, hand="fist", hand_ang=0.8, skip=0)


def p_fiddle(cv, m=None, ox=0):         # 無視しないで
    pen = Pen(cv, Xf(tx=213 + ox, ty=200, sx=1.15, sy=1.15))
    torso(pen, hem=520)
    head(pen, "half", _mouth(m, "o"), tilt=9, look=(0, 6))
    sleeve(pen, [(-62, 200), (-96, 300), (-60, 360), (-10, 352)], 58, 62, hand="fist", hand_ang=-0.2)
    sleeve(pen, [(62, 200), (96, 300), (60, 360), (14, 350)], 58, 62, hand="fist", hand_ang=0.2)


def p_away(cv, m=None, ox=0):           # 謝るからもう
    pen = Pen(cv, Xf(tx=213 + ox, ty=230, sx=1.0, sy=1.0))
    torso(pen, hem=520)
    sleeve(pen, [(-62, 200), (-92, 320), (-96, 450)], 58, 62)
    sleeve(pen, [(62, 200), (92, 320), (96, 450)], 58, 62)
    head(pen, "side", _mouth(m, "line"), tilt=-12, look=(-9, 2))


def p_pinch(cv, m=None, ox=0):          # 置いてかないで — пальцы держат чужой рукав
    pen = Pen(cv, Xf(tx=ox, ty=0, sx=1.0, sy=1.0))
    pen.shape([(-40, -40), (260, -40), (180, 300), (60, 470), (-40, 520)], DARK)
    pen.line([(40, 120), (120, 220)], 0.6)
    s = 2.0
    pen.T = Xf(tx=420 + ox, ty=640, sx=s, sy=s)
    sleeve(pen, [(60, 60), (-30, -30), (-110, -110)], 92, 96, hand="pinch", hand_ang=0.1, skip=0)


def p_worried(cv, m=None, ox=0):        # 嫌いになってない？
    pen = Pen(cv, Xf(tx=213 + ox, ty=250, sx=1.0, sy=1.0))
    torso(pen, hem=520)
    sleeve(pen, [(-62, 200), (-92, 320), (-90, 450)], 58, 62)
    sleeve(pen, [(62, 200), (92, 320), (90, 450)], 58, 62)
    head(pen, "worried", _mouth(m, "o"))
    pen.shape(catmull([(-24, 70), (-20, 84), (-24, 92), (-28, 84)], closed=True), hexc("#cdeaff"), 0.7)


def p_smug(cv, m=None, ox=0):           # だよねw
    pen = Pen(cv, Xf(tx=213 + ox, ty=240, sx=1.0, sy=1.0))
    torso(pen, hem=520)
    sleeve(pen, [(62, 200), (92, 320), (90, 450)], 58, 62)
    head(pen, "half", _mouth(m, "w"), tilt=-7)
    sleeve(pen, [(-80, 340), (-66, 230), (-30, 120)], 60, 62, hand="fist", hand_ang=-0.8, skip=0)


# ---------------- танец (полный рост, на всю сцену) ----------------

def dance(cv, k, x, m=None, eyes="happy"):
    pen = Pen(cv, Xf(tx=x, ty=150, sx=0.5, sy=0.5))
    mo = _mouth(m, "grin")
    if k == 0:      # рука вверх, вторая в сторону, мах ногой
        legs(pen, left=(-6, 0), right=(38, 12))
        torso(pen)
        sleeve(pen, [(-70, 190), (-150, 196), (-236, 200)], 58, 62, hand="open")
        head(pen, eyes, mo, tilt=-6)
        sleeve(pen, [(70, 190), (112, 100), (140, -10)], 58, 62, hand="fist")
    elif k == 1:    # «салют»: рукав ко лбу, другой — указывает в сторону
        legs(pen, left=(-34, -8), right=(4, 0))
        torso(pen)
        sleeve(pen, [(70, 190), (160, 200), (250, 196)], 58, 62, hand="fist")
        head(pen, "xx", mo, tilt=8)
        sleeve(pen, [(-70, 190), (-120, 130), (-90, 30), (-50, -20)], 58, 62, hand="open", hand_ang=-0.4)
    elif k == 2:    # хихикает, рукава у рта
        legs(pen, left=(4, -6), right=(-4, 6))
        torso(pen)
        head(pen, "happy", "w", tilt=-4)
        sleeve(pen, [(-70, 190), (-100, 270), (-50, 210), (-16, 120)], 58, 62, hand="fist", hand_ang=-0.7)
        sleeve(pen, [(70, 190), (100, 270), (50, 210), (18, 124)], 58, 62, hand="fist", hand_ang=0.7)
    else:           # «самолётик», нога согнута
        legs(pen, left=(-10, 0), right=(50, -80))
        torso(pen)
        sleeve(pen, [(-70, 190), (-150, 170), (-236, 140)], 58, 62, hand="open")
        sleeve(pen, [(70, 190), (150, 210), (236, 240)], 58, 62, hand="open")
        head(pen, eyes, mo, tilt=10)


PANEL_POSES = dict(lying=p_lying, chin=p_chin, knees=p_knees, close_mouth=p_close_mouth, fists=p_fists,
                   shrug=p_shrug, cheeks=p_cheeks, hand_mouth=p_hand_mouth, fiddle=p_fiddle, away=p_away,
                   pinch=p_pinch, worried=p_worried, smug=p_smug,
                   close_cheeks=lambda cv, m=None, ox=0: p_cheeks(cv, m, ox, eyes="half", zoom=2.0, mouth="w"),
                   close_pout=lambda cv, m=None, ox=0: p_cheeks(cv, m, ox, eyes="xx", zoom=2.0, mouth="pout"))
