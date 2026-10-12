"""Сцены клипа: фон, поза, типографика и эффекты на каждую строку припева."""
import math
import numpy as np
from flat import (Canvas, Tex, Xf, catmull, ellipse, capsule, hexc, LINE, W, H, SS)
import chara
from chara import C
import lyrics as L

FPS = 24
TWOS = 2

RED = hexc("#e3203f")
RED_D = hexc("#6e0c22")
BLUE = hexc("#2448ff")
BLUE_D = hexc("#0d1150")
PURPLE = hexc("#3a1660")
PINK = hexc("#ff8fb0")
YELLOW = hexc("#ffe14a")
WHITE = np.array([1.0, 1.0, 1.0])
BLACK = hexc("#0b0710")

B = L.BEATS


def snap(abs_t):
    t = L.ct(abs_t)
    return float(B[np.argmin(np.abs(B - t))]) if t > 0 else 0.0


def ease(x):
    x = min(1.0, max(0.0, x))
    return 1 - (1 - x) ** 3


def beat_idx(t):
    return L.beat_phase(t)[0]


# ---------------- фоновые элементы ----------------

def bg_split(cv, c1, c2, slope=0.35, x=W / 2):
    cv.img[:] = np.asarray(c1, np.float32)
    cv.fill(np.array([(x + H * slope / 2, -20), (W + 20, -20), (W + 20, H + 20), (x - H * slope / 2, H + 20)]), c2, jitter=False)


def rings(cv, cx, cy, n, c1, c2, r0=60, step=55, pulse=0.0):
    for k in range(n, 0, -1):
        r = r0 + step * k + pulse * 12 * k / n
        cv.fill(ellipse(cx, cy, r, r, 0, 90), c1 if k % 2 else c2, jitter=False)


def spiral(cv, cx, cy, rot, c1, c2, arms=6):
    cv.img[:] = np.asarray(c1, np.float32)
    for a in range(arms):
        base = rot + a * 2 * math.pi / arms
        pts_in, pts_out = [], []
        for r in np.linspace(5, 1100, 70):
            th = base + r / 150
            wdt = math.pi / arms * 0.95
            pts_in.append((cx + r * math.cos(th), cy + r * math.sin(th)))
            pts_out.append((cx + r * math.cos(th + wdt), cy + r * math.sin(th + wdt)))
        cv.fill(np.array(pts_in + pts_out[::-1]), c2, jitter=False)


def pixel_heart(cv, x, y, s, color):
    bm = [".XX.XX.", "XXXXXXX", "XXXXXXX", ".XXXXX.", "..XXX..", "...X..."]
    for j, row in enumerate(bm):
        for i, ch in enumerate(row):
            if ch == "X":
                cv.rect(x + i * s, y + j * s, x + (i + 1) * s, y + (j + 1) * s, color)


def sparkle(cv, x, y, r, color, key=0, w=0):
    P = np.array([(x, y - r), (x + r * 0.22, y - r * 0.22), (x + r, y), (x + r * 0.22, y + r * 0.22),
                  (x, y + r), (x - r * 0.22, y + r * 0.22), (x - r, y), (x - r * 0.22, y - r * 0.22)])
    cv.fill(P, color)
    if w:
        cv.stroke(P, w, closed=True, key=key)


def curtains(cv, t, base=hexc("#b8142e")):
    cv.img[:] = np.asarray(base * 0.6, np.float32)
    n = 14
    for i in range(n + 1):
        x = i * W / n + 6 * math.sin(t * 1.3 + i)
        for k, (dx, f) in enumerate([(-W / n / 2, 0.55), (-W / n / 4, 0.85), (0, 1.0), (W / n / 5, 0.75)]):
            cv.fill(np.array([(x + dx, -10), (x + dx + W / n / 3.5, -10), (x + dx + W / n / 3.5 + 4, H + 10), (x + dx + 4, H + 10)]),
                    base * f, jitter=False)


def tarot_frame(cv, x0, y0, x1, y1, title, key=0):
    gold = hexc("#d9a640")
    for k, (pad, w) in enumerate([(0, 4.0), (12, 2.0)]):
        P = np.array([(x0 + pad, y0 + pad), (x1 - pad, y0 + pad), (x1 - pad, y1 - pad), (x0 + pad, y1 - pad)])
        cv.stroke(P, w, color=gold, closed=True, key=key + k)
    for cx, cy in ((x0, y0), (x1, y0), (x0, y1), (x1, y1)):
        sx, sy = (1 if cx == x0 else -1), (1 if cy == y0 else -1)
        cv.stroke(catmull([(cx + sx * 12, cy + sy * 50), (cx + sx * 30, cy + sy * 30), (cx + sx * 50, cy + sy * 12)]), 2.4, color=gold, key=key + 5)
        sparkle(cv, cx + sx * 26, cy + sy * 26, 9, gold)
    cv.rect(x0 + 40, y1 - 64, x1 - 40, y1 - 24, hexc("#2a1030"))
    cv.stroke(np.array([(x0 + 40, y1 - 64), (x1 - 40, y1 - 64), (x1 - 40, y1 - 24), (x0 + 40, y1 - 24)]), 2.0, color=gold, closed=True, key=key + 9)
    cv.text(title, (x0 + x1) / 2, y1 - 44, 24, "dotgothic16", gold, anchor="c")
    cv.text("0", (x0 + x1) / 2, y0 + 38, 28, "dotgothic16", gold, anchor="c")


def hp_bar(cv, x, y, w, frac, label, color, right=False):
    cv.text(label, x - 8 if not right else x + w + 8, y + 12, 26, "dotgothic16", WHITE, LINE, 2, anchor="r" if not right else "l")
    cv.rect(x - 3, y - 3, x + w + 3, y + 27, WHITE)
    cv.rect(x, y, x + w, y + 24, BLACK)
    n = 12
    filled = int(round(frac * n))
    for i in range(n):
        j = (n - 1 - i) if right else i
        if i < filled:
            c = color * (0.75 + 0.25 * (i / n))
            cv.rect(x + j * w / n + 2, y + 2, x + (j + 1) * w / n - 2, y + 22, c)


def static_noise(cv, seed, amt=0.6):
    r = np.random.default_rng(seed)
    n = r.random((H // 3 + 1, W // 3 + 1)).repeat(3, 0).repeat(3, 1)[:H, :W]
    n = np.kron(n, np.ones((SS, SS)))[:cv.img.shape[0], :cv.img.shape[1]]
    cv.img[:] = cv.img * (1 - amt) + n[..., None] * amt * 0.35


def stamp(cv, s, x, y, px, color, t_since, rot=0.0, fam="dela-gothic-one", ow=7, out=LINE):
    """Надпись-штамп: влетает с увеличением."""
    if t_since < 0:
        return
    sc = 1.0 + 0.6 * math.exp(-t_since * 18)
    cv.text(s, x, y, px, fam, color, out, ow, anchor="c", rot=rot, scale=sc)


def smoke(cv, t, x, y, key):
    for k in range(3):
        ph = t * 0.8 + k * 0.7
        pts = [(x + 18 * math.sin(ph * 2 + j * 0.9) + j * 3, y - j * 26 - (ph % 1) * 20) for j in range(9)]
        cv.stroke(catmull(pts), 10 - k * 2, color=hexc("#e9e2f2"), key=key + k, taper=(0.3, 0.6))


# ---------------- поза по умолчанию ----------------

def base_pose(t, energy=1.0):
    i, ph = L.beat_phase(t)
    pulse = math.exp(-ph * 3.5)
    side = 1 if (i % 2) else -1
    v, op = L.mouth_at(t)
    p = dict(
        head_rot=3.5 * energy * side * (1 - ph) + 1.5 * math.sin(t * 1.3),
        hy=-7 * energy * pulse, hx=2 * energy * side,
        sway=6 * energy * math.sin(math.pi * (i + ph)),
        aho=t * 9, tie=3 * math.sin(t * 5),
        eyes="open", brow="neutral",
        mouth=dict(kind="sing", vowel=v, open=op),
        blush=0.6,
    )
    # моргание
    for bt in (3.0, 7.6, 12.1, 18.4, 22.0, 25.2):
        if 0 <= t - bt < 0.17:
            p["eyes"] = "closed"
    return p, pulse


def root_xf(cx, cy, s, pulse=0.0, zoom=0.03):
    k = s * (1 + zoom * pulse)
    return Xf(tx=cx, ty=cy, sx=k, sy=k)


# ---------------- сцены ----------------

def sc_traffic(cv, t, lt, fx):
    cv.gradient(hexc("#0a0f3a"), hexc("#141a63"))
    for x in range(0, W, 64):
        cv.rect(x, 0, x + 2, H, hexc("#1f2a8a"), 0.6)
    for y in range(0, H, 64):
        cv.rect(0, y, W, y + 2, hexc("#1f2a8a"), 0.6)
    # светофор
    hx, hy = 1010, 230
    cv.shape(np.array([(hx - 90, hy - 190), (hx + 90, hy - 190), (hx + 90, hy + 190), (hx - 90, hy + 190)]), hexc("#1d1b26"), 4)
    glow = 0.75 + 0.25 * math.sin(t * 7)
    for r, a in ((230, 0.08), (170, 0.14), (125, 0.22), (90, 0.4)):
        cv.fill(ellipse(hx, hy - 105, r, r), RED, alpha=a * glow, jitter=False)
    cv.shape(ellipse(hx, hy - 105, 62, 62), RED * (0.8 + 0.2 * glow) + 0.15, 4)
    cv.fill(ellipse(hx - 18, hy - 125, 16, 12, -0.5), WHITE, alpha=0.7)
    cv.shape(ellipse(hx, hy + 20, 62, 62), hexc("#3a2a1a"), 4)
    cv.halftone(RED, 10, 0.5, density=lambda x, y: np.clip(1 - np.hypot(x - hx, y - hy + 105) / 420, 0, 1) * 0.7)
    p, pulse = base_pose(t, 0.4)
    p.update(eyes="half", brow="sad", head_rot=-4 + 1.5 * math.sin(t * 2), blush=0.3,
             look=(6, 0), armL=None, armR=None)
    z = 1.25 + 0.06 * lt
    chara.draw(cv, p, root_xf(500, 290, z))
    cv.halftone(BLUE_D, 8, 0.8, density=lambda x, y: np.clip((x - 700) / 600, 0, 1) * 0.35)
    fx["tint"] = (1.0, 0.9, 1.05)


def sc_vs(cv, t, lt, fx):
    i, ph = L.beat_phase(t)
    bg_split(cv, RED, BLUE_D, slope=0.5)
    pulse = math.exp(-ph * 3)
    rings(cv, 640, 300, 7, BLACK, WHITE, r0=40, step=58, pulse=pulse)
    cv.halftone(RED_D, 11, 0.6, density=lambda x, y: np.clip(1 - x / 900, 0, 1) * 0.6)
    p, pulse = base_pose(t)
    at = t + L.CLIP_T0
    p.update(brow="smug", eyes="open")
    if at >= 45.25 and at < 45.95:          # «アナタ» — на зрителя
        p["armR"] = dict(target=(150, 175), hand="point", hrot=-35, bend=1)
        p["look"] = (0, 2)
    elif at >= 45.95:                       # «アタシ» — на себя
        p["armL"] = dict(target=(-20, 230), hand="point", hrot=95, bend=-1)
        p["eyes"] = "half"
    chara.draw(cv, p, root_xf(640, 255, 1.35, pulse))


def sc_dai(cv, t, lt, fx):
    i, ph = L.beat_phase(t)
    cv.img[:] = np.asarray(BLACK, np.float32)
    flash = (i % 2 == 0)
    cv.fill(np.array([(0, 0), (W, 0), (W, H), (0, H)]), (RED_D if flash else BLUE_D), jitter=False)
    cv.halftone(RED if flash else BLUE, 13, 0.4, density=lambda x, y: np.clip(np.hypot(x - 640, y - 330) / 700, 0, 1))
    p, pulse = base_pose(t, 1.6)
    p.update(eyes="angry", brow="angry", blush=0.2)
    if L.mouth_at(t)[1] > 0.15:
        p["mouth"] = dict(kind="grin", open=L.mouth_at(t)[1])
    chara.draw(cv, p, root_xf(640, 300, 1.75, pulse, 0.06))
    # штампы «ダイ» на каждую долю
    t0 = snap(46.4)
    k0 = int(np.argmin(np.abs(B - t0)))
    pos = [(230, 170, -0.15), (1050, 160, 0.12), (200, 470, 0.1), (1080, 460, -0.12)]
    for k in range(4):
        tb = B[k0 + k] if k0 + k < len(B) else 99
        if t >= tb:
            x, y, r = pos[k]
            stamp(cv, "ダイ", x, y, 130, WHITE if k % 2 == 0 else YELLOW, t - tb, r, out=RED_D if k % 2 else BLUE_D)
    tb = B[k0 + 4] if k0 + 4 < len(B) else 99
    if t >= tb:
        stamp(cv, "ダイキライ", 640, 560, 150, RED, t - tb, -0.05, out=WHITE, ow=9)
    fx["shake"] = (np.random.default_rng(i).normal(0, 6) * pulse, np.random.default_rng(i + 99).normal(0, 6) * pulse)
    fx["ca"] = 1 + 3 * pulse


def sc_omg(cv, t, lt, fx):
    cv.gradient(hexc("#5d6587"), hexc("#2c3255"))
    cv.halftone(hexc("#1b1f3d"), 9, 0.3, density=lambda x, y: (y / H) * 0.8)
    for k in range(9):
        x = (k * 157 + 40) % W
        y = ((t * 380 + k * 211) % (H + 100)) - 50
        cv.stroke(np.array([(x, y), (x - 6, y + 34)]), 3, color=hexc("#a9c4ff"), key=k)
    p, pulse = base_pose(t, 0.6)
    p.update(eyes="half", brow="sad", mouth=dict(kind="flat") if L.mouth_at(t)[1] < 0.15 else p["mouth"],
             sweat=[(78, 10)], blush=0.0, head_rot=-6,
             armL=dict(target=(-200, 180), hand="open", hrot=-70, bend=1),
             armR=dict(target=(200, 180), hand="open", hrot=70, bend=-1))
    chara.draw(cv, p, root_xf(640, 255, 1.3, pulse))
    stamp(cv, "OMG", 220, 140, 110, YELLOW, lt, -0.2, fam="dotgothic16", ow=6)


def sc_bye(cv, t, lt, fx):
    i, ph = L.beat_phase(t)
    cv.gradient(hexc("#ffb3c9"), hexc("#ff6f9a"))
    for k in range(14):
        x = (k * 97 + t * 60 * (1 if k % 2 else -1)) % (W + 80) - 40
        y = (k * 53) % H
        pixel_heart(cv, x, y, 6, hexc("#ff3f7a") if k % 3 else WHITE)
    p, pulse = base_pose(t, 1.2)
    w = math.sin(math.pi * 2 * (i + ph) / 2)   # машет на каждую «バイ»
    p.update(eyes="happy", brow="up",
             armR=dict(target=(170 + 25 * w, 40), hand="open", hrot=20 * w, bend=1))
    if L.mouth_at(t)[1] > 0.15:
        p["mouth"] = dict(kind="grin", open=0.8)
    chara.draw(cv, p, root_xf(600, 255, 1.3, pulse))
    stamp(cv, "BYE", 1000, 150, 100, WHITE, lt, 0.15, out=hexc("#c3124a"))
    for k in range(5):
        sparkle(cv, 820 + 70 * math.cos(t * 2 + k * 1.3), 380 + 60 * math.sin(t * 3 + k), 10 + 4 * math.sin(t * 6 + k), WHITE)


def sc_spiral(cv, t, lt, fx):
    spiral(cv, 640, 330, -t * 2.2, RED_D, RED, 6)
    p, pulse = base_pose(t, 0.8)
    p.update(eyes="spiral", brow="up", blush=0.8)
    chara.draw(cv, p, root_xf(640, 330, 2.3 + 0.1 * lt, pulse, 0.05))
    i, ph = L.beat_phase(t)
    fx["glitch"] = 0.6 if ph < 0.18 and i % 2 == 0 else 0.0


def sc_kai(cv, t, lt, fx):
    cv.gradient(PURPLE, BLACK)
    cv.halftone(hexc("#6a2aa0"), 12, 0.9, density=lambda x, y: 1 - y / H)
    t0 = snap(53.7)
    k0 = int(np.argmin(np.abs(B - t0)))
    pos = [(180, 130), (420, 90), (870, 100), (1110, 150), (160, 420), (1120, 430)]
    for k, (x, y) in enumerate(pos):
        tb = t0 + k * (B[1] - B[0]) * 0.5
        if t >= tb:
            stamp(cv, "解", x, y, 120, WHITE if k % 2 else YELLOW, t - tb, (k - 2.5) * 0.08, out=PURPLE)
    p, pulse = base_pose(t, 0.7)
    p.update(eyes="shock", brow="sad", sweat=[(-80, 0), (86, 20)], blush=0.0,
             armR=dict(target=(46, 76), hand="open", hrot=60, bend=1))
    if L.mouth_at(t)[1] < 0.15:
        p["mouth"] = dict(kind="wavy")
    chara.draw(cv, p, root_xf(640, 270, 1.35, pulse))
    if t > t0 + 1.2:
        stamp(cv, "は無い", 640, 520, 90, RED, t - t0 - 1.2, 0, out=WHITE)


def sc_curtain(cv, t, lt, fx):
    curtains(cv, t)
    cv.fill(np.array([(520, -10), (760, -10), (980, H + 10), (300, H + 10)]), hexc("#ffd9a0"), alpha=0.22, jitter=False)
    p, pulse = base_pose(t, 0.9)
    at = t + L.CLIP_T0
    p.update(eyes="half", brow="smug", head_rot=8 + 2 * math.sin(t * 3))
    if at >= 56.6:
        p["armR"] = dict(target=(70, 140), hand="point", hrot=-40, bend=1)
        p["eyes"] = "open"
    else:
        p["armL"] = dict(target=(-60, 115), hand="fist", hrot=20, bend=-1)
    chara.draw(cv, p, root_xf(640, 270, 1.4, pulse))
    stamp(cv, "？", 1040, 200, 160, YELLOW, lt - 0.3, 0.2)


def sc_game(cv, t, lt, fx):
    i, ph = L.beat_phase(t)
    cv.gradient(hexc("#1a0d2e"), hexc("#3b0f2a"))
    for x in range(0, W, 40):
        cv.rect(x, 520, x + 38, 560, hexc("#5a1a3a") if (x // 40) % 2 else hexc("#4a1230"))
    cv.rect(0, 560, W, H, hexc("#2a0a1e"))
    cv.halftone(RED, 9, 0.7, density=lambda x, y: np.clip((y - 300) / 400, 0, 1) * 0.5)
    t0 = snap(57.5)
    k = max(0, sum(1 for b in B if t0 <= b <= t))
    hp2 = max(0.0, 1 - min(k, 5) * 0.19)
    brk = L.ct(58.3) <= t < L.ct(59.9)
    p, pulse = base_pose(t, 0.3 if brk else 1.4)
    p.update(eyes="angry", brow="angry")
    if L.mouth_at(t)[1] > 0.15:
        p["mouth"] = dict(kind="grin", open=L.mouth_at(t)[1])
    if brk:
        p.update(eyes="open", brow="up", head_rot=10, mouth=dict(kind="sing", vowel=L.mouth_at(t)[0], open=L.mouth_at(t)[1]))
        p["armR"] = dict(target=(110, 120), hand="peace", hrot=-10, bend=1)
    chara.draw(cv, p, root_xf(640, 290, 1.25, pulse))
    hp_bar(cv, 120, 40, 420, 0.92, "1P", RED)
    hp_bar(cv, 740, 40, 420, hp2, "2P", BLUE, right=True)
    cv.text("VS", 640, 52, 44, "dela-gothic-one", YELLOW, LINE, 5, anchor="c")
    if brk:
        stamp(cv, "？", 1000, 330, 200, WHITE, t - L.ct(58.3), 0.15, out=BLUE)
    if not brk and ph < 0.15 and k > 0:
        fx["flash"] = 0.25
        fx["flash_color"] = (1, 0.3, 0.4)


def sc_sigh(cv, t, lt, fx):
    cv.gradient(hexc("#060a2c"), hexc("#1a2a8a"))
    r = np.random.default_rng(3)
    for k in range(40):
        x = r.uniform(0, W)
        y = (r.uniform(0, H) + t * 40) % H
        s = 3 if k % 3 else 5
        cv.rect(x, y, x + s, y + s, WHITE, 0.8)
    cv.fill(ellipse(1050, 140, 70, 70), hexc("#fff4c8"), jitter=False)
    cv.fill(ellipse(1075, 125, 62, 62), hexc("#0a1040"), jitter=False)
    p, pulse = base_pose(t, 0.3)
    p.update(eyes="half", brow="sad", look=(-8, 4), head_rot=-9, blush=0.2)
    chara.draw(cv, p, root_xf(560, 270, 1.35 + 0.03 * lt, 0))
    fx["tint"] = (0.85, 0.9, 1.1)


def sc_nothing(cv, t, lt, fx):
    cv.img[:] = np.asarray(BLACK, np.float32)
    p, pulse = base_pose(t, 0.3)
    p.update(eyes="closed", brow="sad",
             armL=dict(target=(-46, 76), hand="open", hrot=60, bend=-1),
             armR=dict(target=(46, 76), hand="open", hrot=60, bend=1))
    chara.draw(cv, p, root_xf(640, 280, 1.3, 0))
    static_noise(cv, int(t * 12), 0.28)
    cv.text("NOTHING", 250, 300, 70, "dotgothic16", WHITE, LINE, 4, anchor="c")
    cv.text("NOT SEE", 1030, 300, 70, "dotgothic16", RED, LINE, 4, anchor="c")
    fx["glitch"] = 0.15


def sc_sunset(cv, t, lt, fx):
    cv.gradient(hexc("#ff9fb7"), hexc("#ff7a3d"))
    cv.fill(ellipse(640, 470, 260, 260), hexc("#fff0a8"), alpha=0.85, jitter=False)
    for k in range(6):
        cv.rect(0, 380 + k * 28, W, 380 + k * 28 + 6 + k * 2, hexc("#ff9fb7"), 0.9)
    cv.halftone(hexc("#c23a6b"), 10, 0.5, density=lambda x, y: np.clip(1 - y / 400, 0, 1) * 0.5)
    tarot_frame(cv, 390, 18, 890, 600, "THE FOOL", key=500)
    p, pulse = base_pose(t, 0.5)
    p.update(eyes="half", brow="neutral", head_rot=6 * math.sin(t * 1.4), blush=0.8,
             armL=dict(target=(-80, 150), hand="peace", hrot=40, bend=-1))
    chara.draw(cv, p, root_xf(640, 260, 1.2, pulse))
    smoke(cv, t, 820, 430, 600)


def sc_reach(cv, t, lt, fx):
    cv.gradient(hexc("#ff7aa5"), hexc("#7a1d6b"))
    rings(cv, 640, 320, 9, hexc("#ff9cc0"), hexc("#ff7aa5"), r0=30, step=60, pulse=lt * 4 % 1)
    p, pulse = base_pose(t, 0.5)
    p.update(eyes="open", brow="sad", look=(0, 3), blush=1.0,
             armR=dict(target=(40, 200), hand="open", hrot=180, hscale=1.7, bend=1))
    z = 1.3 + 0.35 * ease(lt / 2.4)
    chara.draw(cv, p, root_xf(640, 270 + 20 * ease(lt / 2.4), z, 0))
    stamp(cv, "I miss you", 300, 150, 64, WHITE, lt - 0.1, -0.04, fam="dotgothic16", out=hexc("#7a1d6b"), ow=5)


def sc_bon(cv, t, lt, fx):
    i, ph = L.beat_phase(t)
    second = t >= L.ct(70.05)
    cv.img[:] = np.asarray(YELLOW if not second else hexc("#33e0ff"), np.float32)
    for k in range(12):
        a = k * math.pi / 6 + t
        cv.fill(np.array([(640, 330), (640 + 1200 * math.cos(a), 330 + 1200 * math.sin(a)),
                          (640 + 1200 * math.cos(a + 0.22), 330 + 1200 * math.sin(a + 0.22))]), WHITE, alpha=0.45, jitter=False)
    p, pulse = base_pose(t, 1.2)
    p.update(eyes="shock", brow="up", sweat=[(-84, -6), (88, 10), (96, 50)], blush=0.0)
    if L.mouth_at(t)[1] < 0.15:
        p["mouth"] = dict(kind="wavy")
    chara.draw(cv, p, root_xf(640 + (40 if second else -40), 270, 1.4, pulse))
    stamp(cv, "凡ミス", 300 if not second else 980, 520, 110, RED, t - (L.ct(70.05) if second else L.ct(69.5)), -0.12 if not second else 0.12, out=WHITE)
    fx["shake"] = (8 * pulse * (1 if i % 2 else -1), 0)


def sc_thanks(cv, t, lt, fx):
    cv.gradient(hexc("#8fe6ff"), hexc("#3d63ff"))
    for k in range(10):
        sparkle(cv, (k * 131 + 60) % W, (k * 89 + 40 + t * 30) % (H - 120), 10 + 6 * abs(math.sin(t * 4 + k)), WHITE)
    p, pulse = base_pose(t, 1.0)
    p.update(eyeR="wink", eyeL="open", brow="smug", mouth=dict(kind="tongue"), head_rot=-8,
             armL=dict(target=(-120, 100), hand="peace", hrot=-20, bend=-1))
    if L.mouth_at(t)[1] > 0.3 and t < L.ct(71.3):
        p["mouth"] = dict(kind="sing", vowel=L.mouth_at(t)[0], open=L.mouth_at(t)[1])
    chara.draw(cv, p, root_xf(640, 260, 1.35, pulse))
    stamp(cv, "THANK YOU", 1010, 150, 64, WHITE, lt, 0.1, fam="dotgothic16", out=BLUE, ow=5)


def sc_end(cv, t, lt, fx):
    curtains(cv, t, hexc("#7a0c22"))
    p, pulse = base_pose(t, 0.4)
    p.update(eyes="half", brow="smug", mouth=dict(kind="smirk") if L.mouth_at(t)[1] < 0.15 else p["mouth"])
    chara.draw(cv, p, root_xf(640, 260, 1.45 + 0.1 * lt, 0))
    fx["glitch"] = 0.2 + 0.8 * ease(lt / 0.7)
    fx["tint"] = tuple(np.array([1.0, 0.85, 0.9]) * (1 - 0.85 * ease((lt - 0.2) / 0.5)))


SCENES = [
    (0.0, sc_traffic),
    (snap(44.2), sc_vs),
    (snap(46.4), sc_dai),
    (snap(48.2), sc_omg),
    (snap(49.8), sc_bye),
    (snap(51.9), sc_spiral),
    (snap(53.7), sc_kai),
    (snap(55.5), sc_curtain),
    (snap(57.5), sc_game),
    (snap(59.9), sc_sigh),
    (snap(62.3), sc_nothing),
    (snap(63.3), sc_sunset),
    (snap(67.1), sc_reach),
    (snap(69.5), sc_bon),
    (snap(70.6), sc_thanks),
    (snap(72.3), sc_end),
]


def subtitles(cv, t):
    k, line = L.line_at(t)
    if line is None:
        return
    a, b, segs, _, en = line
    cv.ruby_line(segs, 640, 640, 30, color=WHITE, outline=LINE, ow=3)
    cv.text(en, 640, 688, 19, "dotgothic16", hexc("#ffe9f0"), LINE, 2.5, anchor="c")


_tex = None


def render(frame):
    global _tex
    if _tex is None:
        _tex = Tex()
    d = frame // TWOS
    t = d * TWOS / FPS
    cv = Canvas(seed=500 + d)
    k = max(i for i, (s, _) in enumerate(SCENES) if t >= s)
    s0, fn = SCENES[k]
    fx = dict(seed=d, grain_i=d, ca=1.2)
    fn(cv, t, t - s0, fx)
    # глитч на каждой склейке
    if k > 0 and t - s0 < 0.09:
        fx["glitch"] = max(fx.get("glitch", 0), 0.9)
    subtitles(cv, t)
    return cv.finish(_tex, fx)


if __name__ == "__main__":
    import sys
    tt = float(sys.argv[1])
    render(int(round(tt * FPS))).save(sys.argv[2])
