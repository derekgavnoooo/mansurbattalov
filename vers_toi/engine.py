"""Движок сборки «Vers toi»: 2.5D-фоны, вырезанные персонажи с ригом, процедурные глаза и эффекты.
Запуск:
  python engine.py <assets> <out.mp4> <audio.wav>            # весь ролик
  python engine.py <assets> <sheet.png> --frames 3,10.5,20    # проверочные кадры
Сборка идёт в 360×640, итог 180×320 с дизером, увеличение ×6 без сглаживания.
"""
import sys, math, subprocess
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 360, 640, 12
OW, OH = 180, 320
GOLD = np.array([[10, 6, 4], [52, 32, 18], [128, 88, 46], [214, 170, 96], [252, 240, 214]], np.float32)
NIGHT = np.array([[3, 4, 9], [16, 20, 34], [44, 52, 74], [112, 120, 140], [222, 228, 236]], np.float32)
BAY = (np.array([0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5]).reshape(4, 4) + .5) / 16
BAYT = np.tile(BAY, (OH // 4, OW // 4))
YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)

# ---------- шум ----------
def h2(x, y, s=0):
    h = (x.astype(np.int64) * 374761393 + y.astype(np.int64) * 668265263 + s * 1442695041) & 0xffffffff
    h = ((h ^ (h >> 13)) * 1274126177) & 0xffffffff
    return ((h ^ (h >> 16)) & 0xffffffff) / 4294967296.0
def vn(x, y, s=0):
    xi, yi = np.floor(x), np.floor(y); xf, yf = x - xi, y - yi
    u, v = xf * xf * (3 - 2 * xf), yf * yf * (3 - 2 * yf)
    a, b, c, d = h2(xi, yi, s), h2(xi + 1, yi, s), h2(xi, yi + 1, s), h2(xi + 1, yi + 1, s)
    return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v
def fbm(x, y, s=0):
    return vn(x, y, s) * .5 + vn(x * 2.03 + 5.2, y * 2.03 + 1.3, s) * .25 + vn(x * 4.01 + 9.1, y * 4.01 + 3.7, s) * .125
def sm(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t)
def rnd(*k):
    return float(h2(np.array(k[0]), np.array(k[1] if len(k) > 1 else 0), k[2] if len(k) > 2 else 0))

# ---------- ассеты ----------
class BG:
    """Фон с картой глубины. Картинка больше кадра (запас), поэтому края никогда не смазываются."""
    def __init__(self, img, dep):
        im = np.asarray(Image.open(img).convert('L'), np.float32) / 255
        d = np.asarray(Image.open(dep).convert('L').resize(im.shape[::-1]), np.float32) / 255
        mh, mw = int(im.shape[0] * .05), int(im.shape[1] * .06)
        im, d = im[mh:-mh, mw:-mw], d[mh:-mh, mw:-mw]
        sw, sh = int(W * 1.22), int(H * 1.22)
        sc = max(sw / im.shape[1], sh / im.shape[0])
        nw, nh = int(im.shape[1] * sc + .5), int(im.shape[0] * sc + .5)
        im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA); d = cv2.resize(d, (nw, nh))
        y0, x0 = (nh - sh) // 2, (nw - sw) // 2
        self.im, self.d = im[y0:y0 + sh, x0:x0 + sw], cv2.GaussianBlur(d[y0:y0 + sh, x0:x0 + sw], (0, 0), 3)
        self.sw, self.sh = sw, sh
    def view(self, zoom=1., px=0., py=0., depth=1., fx=.5, fy=.45):
        """zoom>=1; px,py — сдвиг камеры в долях запаса (-1..1); ближние точки двигаются сильнее."""
        mx, my = (self.sw - W) / 2, (self.sh - H) / 2
        cx, cy = self.sw * fx, self.sh * fy
        ux, uy = XX + mx, YY + my
        sx, sy = ux.copy(), uy.copy()
        for _ in range(3):
            dd = cv2.remap(self.d, sx, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            s = 1 + (zoom - 1) * (.3 + 1.5 * dd ** 2 * depth)
            sx = cx + (ux - cx) / s - px * mx * (.35 + .65 * dd * depth)
            sy = cy + (uy - cy) / s - py * my * (.35 + .65 * dd * depth)
            sx, sy = sx.astype(np.float32), sy.astype(np.float32)
        return cv2.remap(self.im, sx, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

class Sprite:
    """Персонаж, вырезанный с белого фона, с простым ригом крыльев."""
    def __init__(self, path, flip=False):
        g = np.asarray(Image.open(path).convert('L'), np.float32) / 255
        if flip: g = g[:, ::-1].copy()
        cut = path.replace('/ch_', '/cut_')
        try:
            al = np.asarray(Image.open(cut).convert('RGBA'), np.float32)[:, :, 3] / 255
            if flip: al = al[:, ::-1].copy()
            fg = (al > .5).astype(np.uint8)
        except FileNotFoundError:
            bright = (g > .78).astype(np.uint8)
            n, lab = cv2.connectedComponents(bright)
            border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
            fg = (~(np.isin(lab, list(border)) & (bright > 0))).astype(np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(fg)
        if n > 1:
            k = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA]); fg = (lab == k).astype(np.uint8)
        ys, xs = np.nonzero(fg); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        self.g = g[y0:y1, x0:x1]; a = cv2.GaussianBlur(fg[y0:y1, x0:x1].astype(np.float32), (0, 0), 1.2)
        self.a = np.clip(a * 1.4, 0, 1)
        col = self.a.sum(0); self.cx = float((np.arange(len(col)) * col).sum() / col.sum())
        self.h, self.w = self.g.shape
    def draw(self, cv, x, y, height, rot=0., flap=0., lo=.03, hi=.95, alpha=1., rim=0., glow=0.):
        """x,y — центр персонажа на холсте; height — высота в пикселях холста."""
        g, a = self.g, self.a
        if abs(flap) > 1e-3:
            yy, xx = np.mgrid[0:self.h, 0:self.w].astype(np.float32)
            dx = (xx - self.cx) / (self.w * .5)
            mapy = yy + flap * self.h * .12 * np.abs(dx) ** 1.6 * (1 - yy / self.h)
            g = cv2.remap(g, xx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=1)
            a = cv2.remap(a, xx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        s = height / self.h
        M = cv2.getRotationMatrix2D((self.w / 2, self.h / 2), rot, s)
        M[0, 2] += x - self.w / 2; M[1, 2] += y - self.h / 2
        gw = cv2.warpAffine(g, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=1)
        aw = cv2.warpAffine(a, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0) * alpha
        col = lo + (hi - lo) * gw
        if rim > 0:
            edge = np.clip(aw - cv2.GaussianBlur(aw, (0, 0), 2.5), 0, 1) * 3
            col = col + rim * edge
        if glow > 0:
            cv += glow * cv2.GaussianBlur(aw, (0, 0), 14)
        cv[:] = cv * (1 - aw) + col * aw
        return aw

# ---------- процедурные вещи ----------
def eye(cv, cx, cy, r, op=1., gx=0., gy=0., seed=0, t=0., bright=1.):
    """Органический глаз как в «Зрачке», рисуется прямо в холст."""
    x0, x1, y0, y1 = int(max(0, cx - r * 1.5)), int(min(W, cx + r * 1.5)), int(max(0, cy - r * 1.3)), int(min(H, cy + r * 1.3))
    if x1 <= x0 or y1 <= y0 or r < 1.5: return
    dx = (XX[y0:y1, x0:x1] - cx) / r; dy = (YY[y0:y1, x0:x1] - cy) / r; ax = np.abs(dx)
    sh = np.where(ax < 1, np.maximum(0, 1 - dx * dx) ** .8, 0)
    nt, nb = fbm(dx * 2.2 + seed, np.full_like(dx, t * .25)), fbm(dx * 2.2 + seed + 9, np.full_like(dx, t * .2))
    top, bot = -(.58 * op * sh) * (.78 + .44 * nt), (.42 * op * sh) * (.78 + .44 * nb)
    dT, dB = dy - top, bot - dy; ins = np.minimum(dT, dB)
    v = cv[y0:y1, x0:x1]
    out = -ins; fold = np.where(dy < 0, np.abs(out - .17 - .06 * nt), np.abs(out - .13 - .05 * nb))
    v = np.where((ins <= 0) & (ax < 1.3), v - np.exp(-fold * fold / .002) * (1 - ax * .6) * .25 - (ax < 1) * np.exp(-out * out / .002) * .3, v)
    ix, iy = dx - gx, dy - gy; ir = np.hypot(ix, iy); an = np.arctan2(iy, ix)
    Rir = .44 * (1 + .12 * (fbm(an * 1.3 + seed, np.full_like(an, t * .05)) - .5))
    Rp = .15 * (.92 + .2 * (fbm(an * 1.7 + seed, np.full_like(an, t * .3)) - .5))
    vv = fbm(dx * 3 + seed, dy * 3); vein = np.maximum(0, 1 - np.abs(vv - .5) * 9) * ax
    e = .92 - vein * .3 - ax * ax * .3
    q = fbm(an * 5 + fbm(ir * 9 + seed, an * 2) * 1.8 + seed, ir * 12); ib = sm(Rir * 1.08, Rir * .9, ir)
    e = e * (1 - ib) + (.3 + .35 * q) * (.55 + .45 * sm(Rir, Rir * .7, ir)) * ib
    pb = sm(Rp * 1.15, Rp * .88, ir); e = e * (1 - pb) + .02 * pb
    e = np.where((ix + .13) ** 2 + (iy + .15) ** 2 < .007, 1., e)
    e = e * (.45 + .55 * sm(0, .2, dT)) * (.6 + .4 * sm(0, .12, dB)) * bright
    k = sm(0, .05, ins); cv[y0:y1, x0:x1] = np.where(ins > 0, v * (1 - k) + e * k, v)

def ring(cv, cx, cy, R, tilt, spin, t, n_eyes=12, thick=4., bright=1., eyes=True, op=1., look=0., seed=0):
    """Горящее колесо с глазами: неровная полоса из пятен, глаза поверх."""
    for i in range(int(2 * math.pi * R / 3)):
        s = i / (2 * math.pi * R / 3) * 2 * math.pi
        cv2.circle(cv, (int((cx + R * math.cos(s)) * 4), int((cy + R * math.sin(s) * tilt) * 4)), int(thick * 1.9 * 4), .04, -1, cv2.LINE_AA, 2)
    for i in range(int(2 * math.pi * R / 1.6)):
        s = i / (2 * math.pi * R / 1.6) * 2 * math.pi
        rr = R * (1 + .06 * (rnd(int(s * 5) + seed, int(t * 3)) - .5))
        x, y = cx + rr * math.cos(s), cy + rr * math.sin(s) * tilt
        fl = .45 + .55 * float(fbm(np.array(s * 9 + seed), np.array(t * 2.2)))
        cv2.circle(cv, (int(x * 4), int(y * 4)), int(thick * (.7 + .6 * fl) * 4), fl * bright, -1, cv2.LINE_AA, 2)
    if eyes:
        for j in range(n_eyes):
            s = j / n_eyes * 2 * math.pi + spin
            x, y = cx + R * math.cos(s), cy + R * math.sin(s) * tilt
            eye(cv, x, y, thick * 2.1 * (.85 + .3 * rnd(j, seed)), op, (.3 * math.sin(t + j)) * (1 - look), 0, j * 3 + seed, t, bright)

FEATHER = None
def feather_sprite():
    global FEATHER
    if FEATHER is None:
        yy, xx = np.mgrid[0:40, 0:14].astype(np.float32); u = yy / 40; v = (xx - 7) / 7
        w = np.sin(np.pi * u) ** .7 * (1 + .3 * (fbm(u * 6, v) - .5))
        a = ((np.abs(v) < w) & (u > .05)).astype(np.float32); a[np.abs(xx - 7) < .6] = 1
        g = .65 + .25 * np.sin(u * 40 + np.abs(v) * 6) * .5
        FEATHER = (g.astype(np.float32), cv2.GaussianBlur(a, (0, 0), .6))
    return FEATHER
def feathers(cv, t, n, seed, speed=30., size=1., bright=1., up=False):
    g, a = feather_sprite()
    for k in range(n):
        sc = (.4 + 1.4 * rnd(k, seed) ** 2) * size
        y = ((rnd(k, seed + 1) * (H + 80) + (-1 if up else 1) * t * speed * (.5 + sc)) % (H + 80)) - 40
        x = rnd(k, seed + 2) * W + 20 * math.sin(t * .7 + k) * sc
        M = cv2.getRotationMatrix2D((7, 20), math.degrees(1.5 * math.sin(t * .9 + k * 2)) + rnd(k, seed + 3) * 360, sc)
        M[0, 2] += x - 7; M[1, 2] += y - 20
        aw = cv2.warpAffine(a, M, (W, H)); gw = cv2.warpAffine(g, M, (W, H))
        cv[:] = cv * (1 - aw) + gw * bright * aw

def specks(cv, t, n, seed, vy=40., bright=.8, size=1):
    for k in range(n):
        x = int(rnd(k, seed) * W + 6 * math.sin(t * 1.3 + k)); y = int((rnd(k, seed + 1) * H + t * vy * (.5 + rnd(k, seed + 2))) % H)
        cv2.circle(cv, (x, y), size, bright * (.4 + .6 * rnd(k, int(t * 12))), -1)

def speedlines(cv, t, n=40, seed=0, up=True, bright=.5):
    for k in range(n):
        x = int(rnd(k, seed) * W); L = 30 + 90 * rnd(k, seed + 1)
        y = ((rnd(k, seed + 2) * (H + L) - (1 if up else -1) * t * 900 * (.5 + rnd(k, seed + 3))) % (H + L)) - L
        cv2.line(cv, (x, int(y)), (x, int(y + L)), bright * (.3 + .7 * rnd(k, seed + 4)), 1)

def lightning(cv, fr, x0, y0, y1, seed=0):
    x, y = x0, y0
    while y < y1:
        nx = x + (rnd(fr, int(y), seed) - .5) * 26; ny = y + 8 + 14 * rnd(fr, int(y) + 7, seed)
        cv2.line(cv, (int(x), int(y)), (int(nx), int(ny)), 1.3, 2, cv2.LINE_AA); x, y = nx, ny

def rays(cv, lx, ly, strength=.5, thr=.55):
    src = np.clip(cv - thr, 0, None); acc = np.zeros_like(src); cur = src
    for i in range(1, 9):
        s = 1 + i * .045
        M = np.float32([[s, 0, lx * (1 - s)], [0, s, ly * (1 - s)]])
        acc += cv2.warpAffine(src, M, (W, H)) * (1 - i / 10)
    cv += acc * strength / 5

def bloom(cv, k=.35, thr=.6):
    cv += cv2.GaussianBlur(np.clip(cv - thr, 0, None), (0, 0), 9) * k

def glowspot(cv, x, y, r, a):
    cv += a * np.exp(-((XX - x) ** 2 + (YY - y) ** 2) / (2 * r * r))

# ---------- раскадровка ----------
A = {}
def load(assets, pick):
    for k in ['heaven', 'cloudsea', 'storm', 'columns', 'void', 'light', 'dawn', 'edge']:
        A[k] = BG(f'{assets}/bg_{k}_{pick.get(k, 0)}.png', f'{assets}/dep_{k}_{pick.get(k, 0)}.png')
    for k in ['elder', 'younger', 'kneel', 'dive', 'fall', 'embrace', 'back']:
        A[k] = Sprite(f'{assets}/ch_{k}_{pick.get(k, 0)}.png')
        A[k + '_f'] = Sprite(f'{assets}/ch_{k}_{pick.get(k, 0)}.png', flip=True)

def fl(t, sp=2.2, a=.5): return a * math.sin(t * sp)
# каждая сцена: (начало, функция(lt, t, E, fr) -> холст, холод палитры 0..1)
def s_hook(lt, t, E, fr):
    c = A['cloudsea'].view(1 + .12 * lt / 3.5, .2 * math.sin(lt * .5), 0)
    eye(c, 180, 230, 150, sm(.1, .5, lt), 0, 0, 7, t, .9)
    A['elder'].draw(c, 135, 500, 230, 0, fl(t), .05, .9, rim=.3, glow=.2); A['younger_f'].draw(c, 250, 525, 185, 0, fl(t + 1), .04, .7, rim=.25)
    return c
def s_hall(lt, t, E, fr):
    c = A['heaven'].view(1.05 + .1 * lt / 3.5, -.6 + .4 * lt / 3.5, 0)
    glowspot(c, 170, 300, 60, .5 + .3 * E)
    A['elder'].draw(c, 160, 430, 230, 0, fl(t), .05, 1., glow=.25)
    A['younger'].draw(c, 280, 470, 170, 0, fl(t + 1.3), .03, .55, rim=.25)
    return c
def s_young_close(lt, t, E, fr):
    c = A['cloudsea'].view(1.15, .5, .2) * .7
    glowspot(c, 60, 120, 90, .7)
    A['younger'].draw(c, 200 - 8 * lt, 380, 520, -3, fl(t, 1.6, .3), .03, .8, rim=.5)
    return c
def s_elder_halo(lt, t, E, fr):
    c = A['heaven'].view(1.2 + .05 * lt, .2, -.3)
    A['elder'].draw(c, 180, 400, 470, 0, fl(t, 1.6, .3), .04, 1., glow=.35)
    ring(c, 180, 125, 80, .32, t * .5, t, 9, 6.5, 1., True, 1)
    return c
def s_wheels_close(lt, t, E, fr):
    c = A['heaven'].view(1.3, 0, 0) * (1 - .4 * sm(0, 3.5, lt))
    A['elder'].draw(c, 180, 380, 300, 0, fl(t, 1, .2), .04, .9)
    for k, (R, ti) in enumerate([(210 - 40 * lt, .35), (170 - 30 * lt, .8), (250 - 50 * lt, .2)]):
        ring(c, 180, 330, max(60, R), ti, t * (.3 + .2 * k) * (1 if k % 2 else -1), t, 14, 4.5, .9, True, 1, 0, k * 11)
    return c
def s_eyefield(lt, t, E, fr, look=0., turn=0.):
    c = A['heaven'].view(1.1 + .06 * lt, 0, 0) * .55
    for k in range(46):
        x, y = rnd(k, 5) * W, 40 + rnd(k, 6) * 560; r = 8 + 18 * rnd(k, 7) ** 2
        op = sm(rnd(k, 8) * 1.2, rnd(k, 8) * 1.2 + .3, lt) * (1 - turn * sm(.5 + rnd(k, 9) * 2, 1 + rnd(k, 9) * 2, lt))
        eye(c, x, y, r, op, (180 - x) / 400 * look + (1 - look) * .3 * math.sin(t + k) + turn * .4, 0, k, t, .95)
    A['elder'].draw(c, 180, 470, 200, 0, 0, .02, .4, rim=.4)
    return c
def s_turn_eye(lt, t, E, fr):
    c = A['heaven'].view(1.25, 0, 0) * .4
    eye(c, 180, 300, 140, 1 - sm(2.2, 3.4, lt), .1 + .5 * sm(.6, 1.6, lt), 0, 3, t, 1)
    return c
def s_kneel(lt, t, E, fr):
    c = A['heaven'].view(1.1 + .08 * lt, 0, .4)
    rays(c, 180, -40, .9, .45)
    A['kneel'].draw(c, 180, 470, 230, 0, 0, .02, .7, rim=.3)
    glowspot(c, 180, 100, 70, .4 + .3 * E)
    return c
def s_step(lt, t, E, fr):
    c = A['heaven'].view(1.1, -.4 + .3 * lt / 3.5, 0)
    A['kneel'].draw(c, 120, 500, 170, 0, 0, .02, .45)
    A['younger_f'].draw(c, 330 - 70 * sm(0, 3, lt), 470, 250, 0, fl(t, 2, .4), .05, .95, rim=.3, glow=.15)
    return c
def s_tear(lt, t, E, fr):
    c = A['heaven'].view(1.3, 0, 0) * .7
    A['elder'].draw(c, 180, 380, 420, 0, -sm(0, 3, lt) * .8 + fl(t, 7, .1), .04, .9)
    feathers(c, lt * 2, 26, 3, 120, 1.4, 1.)
    return c
def s_strike(lt, t, E, fr):
    c = A['cloudsea'].view(1 + .3 * lt / 4, 0, .3)
    hole = sm(1., 2.5, lt)
    c *= 1 - .9 * hole * np.exp(-((XX - 180) ** 2 + (YY - 520) ** 2 * 2) / (2 * 90 ** 2))
    if lt < .25 or 1.0 < lt < 1.2: c += .8
    lightning(c, fr, 180 + 30 * math.sin(fr), 0, 470, 1) if lt < 1.2 else None
    A['elder'].draw(c, 180, 450 + 140 * sm(2.2, 4, lt) ** 2, 220 * (1 - .5 * sm(2.2, 4, lt)), 20 * sm(2.2, 4, lt), 0, .03, .8, rim=.3)
    return c
def s_fall_away(lt, t, E, fr):
    c = A['storm'].view(1 + .5 * lt / 4, 0, 0, 1.2, .5, .5)
    s = 260 * math.exp(-.7 * lt); A['fall'].draw(c, 180, 330 + 20 * lt, s, lt * 25, fl(t, 5, .4), .02, .8, rim=.4)
    feathers(c, t, 10, 9, -60, .7, .9, True)
    return c
def s_edge(lt, t, E, fr):
    c = A['edge'].view(1.05 + .05 * lt, .3 * math.sin(lt), .2)
    A['back'].draw(c, 180, 330, 360, 0, fl(t, 1.4, .3), .02, .45, rim=.5)
    feathers(c, t, 8, 4, 25, .8, .9)
    return c
def s_dive(lt, t, E, fr):
    c = A['edge'].view(1.1 + .3 * sm(1.5, 4, lt), 0, .8 * sm(1.5, 4, lt))
    A['dive'].draw(c, 180, 300 + 120 * sm(1.4, 4, lt) ** 2, 300, 180 * sm(.8, 1.6, lt), -fl(t, 3, .4) * (1 - sm(.8, 1.6, lt)), .03, .8, rim=.5)
    if lt > 1.5: speedlines(c, t, 30, 2, True, .5 * sm(1.5, 2.5, lt))
    return c
def s_fall_storm(lt, t, E, fr):
    c = A['storm'].view(1 + .9 * lt / 4, .3 * math.sin(t * 2), 0, 1.3, .5, .5)
    if rnd(fr, 77) < .15: lightning(c, fr, 60 + 240 * rnd(fr, 78), 0, 640, 2); c += .2
    speedlines(c, t, 50, 3, True, .45)
    A['dive'].draw(c, 180 + 15 * math.sin(t * 3), 320, 300, 180 + 6 * math.sin(t * 5), 0, .02, .75, rim=.3)
    return c
def s_columns(lt, t, E, fr):
    c = A['columns'].view(1 + .8 * lt / 4, -.8 + .5 * lt, .8 - .4 * lt, 1.4)
    speedlines(c, t, 40, 5, True, .4)
    A['dive'].draw(c, 200, 360, 200, 175, 0, .02, .7, rim=.3)
    return c
def s_reach_far(lt, t, E, fr):
    c = A['storm'].view(1.2 + .3 * lt / 4, 0, 0, 1.2) * .8
    A['fall'].draw(c, 180 + 10 * math.sin(t), 520, 70, lt * 15, fl(t, 4, .3), .02, .6, rim=.5)
    A['dive'].draw(c, 180, 170, 210, 180, 0, .02, .7, rim=.3)
    speedlines(c, t, 30, 6, True, .35)
    return c
def s_vortex(lt, t, E, fr):
    a = np.arctan2(YY - 300, XX - 180); r = np.hypot(XX - 180, (YY - 300) * 1.1)
    c = .05 + .45 * sm(.4, .78, fbm(a * 2.2 + r * .012 + t * .5, r * .014 - t * .5))
    c += .9 * np.exp(-r / 40) * (.6 + .4 * fbm(np.full_like(r, t * 3), a * 2))
    A['fall'].draw(c, 180 + 40 * math.cos(t), 300 + 40 * math.sin(t), 50, t * 40, 0, .02, .5)
    A['dive'].draw(c, 180 + 70 * math.cos(t + .6), 300 + 70 * math.sin(t + .6), 70, 180 + t * 40, 0, .02, .5)
    return c
def s_void(lt, t, E, fr, eyes=True, reach=False):
    c = A['void'].view(1.05 + .05 * lt, 0, -.2 * lt / 5) * .9
    if eyes:
        for k in range(22):
            o = sm(.5 + rnd(k, 44) * 3, 1 + rnd(k, 44) * 3, lt)
            if o > 0: eye(c, rnd(k, 41) * W, rnd(k, 42) * H, 5 + 8 * rnd(k, 43), o, 0, 0, k * 5, t, .7)
    if reach:
        A['fall'].draw(c, 230, 470 - 30 * lt / 5, 90, 15, 0, .02, .55, rim=.3)
        A['dive'].draw(c, 140, 180 + 60 * lt / 5, 110, 160, 0, .02, .6, rim=.3)
    else:
        A['dive'].draw(c, 180, 300, 140, 175 + 4 * math.sin(t), 0, .02, .55, rim=.3)
    return c
def s_last_eye(lt, t, E, fr):
    c = A['void'].view(1.15, 0, 0) * .5
    eye(c, 180, 320, 90, 1 - sm(1.5, 5, lt), 0, 0, 99, t, .8)
    return c
def s_rise_light(lt, t, E, fr, close=0.):
    c = A['light'].view(1 + .25 * lt / 4 + close * .3, 0, -.3, 1.2)
    rays(c, 180, 640, .8, .5)
    gap = 1 - close
    A['fall'].draw(c, 180, 470 - 120 * close, 110 + 80 * close, 10, fl(t, 3, .3), .02, .7, rim=.35)
    A['dive'].draw(c, 180, 150 + 160 * close, 160 + 60 * close, 178, 0, .02, .7, rim=.35)
    feathers(c, t, 10, 12, -40, .8, 1., True)
    return c
def s_meet(lt, t, E, fr):
    c = A['light'].view(1.3 + .2 * lt, 0, 0, 1.3)
    if lt < .3: c += 1.
    rays(c, 180, 600, 1.2, .45)
    A['embrace'].draw(c, 180, 330, 480, 5 * math.sin(t), fl(t, 1.5, .5), .04, 1., rim=.6, glow=.3)
    return c
def s_embrace(lt, t, E, fr):
    c = A['light'].view(1.1 + .3 * lt / 4, .3 * math.sin(lt), 0, 1.3)
    rays(c, 180, 640, 1., .45)
    A['embrace'].draw(c, 180, 300 + 20 * lt, 360 - 30 * lt, 8 * math.sin(t * .7), fl(t, 1.2, .7), .04, 1., rim=.5, glow=.3)
    feathers(c, t, 18, 14, -50, 1., 1., True)
    return c
def s_rose(lt, t, E, fr):
    c = A['light'].view(1 + .4 * lt / 4, 0, 0, 1.2)
    for k, (R, ti) in enumerate([(240, .3), (190, .32), (140, .34), (95, .36)]):
        ring(c, 180, 470, R * (1 + .15 * lt / 4), ti, t * (.2 + .1 * k) * (1 if k % 2 else -1), t, 18 - 3 * k, 4, 1., True, 1, 1, k * 7)
    A['embrace'].draw(c, 180, 250 + 150 * lt / 4, 150 * (1 - .4 * lt / 4), 0, fl(t, 1.2, .5), .03, .8, rim=.3)
    rays(c, 180, 470, 1., .5)
    return c
def s_into(lt, t, E, fr):
    c = A['light'].view(1.3 + .9 * lt / 4, 0, 0, 1.5)
    for k in range(3): ring(c, 180, 330, (120 + 90 * k) * (1 + 1.5 * lt / 4), .4 + .2 * k, t * .6 * (k - 1), t, 16, 5, 1., True, 1, 1, k * 13)
    c += sm(2.5, 4, lt) * 1.2
    return c
def s_white(lt, t, E, fr):
    c = np.full((H, W), 1.1 - .6 * sm(1.5, 4, lt), np.float32)
    feathers(c, t, 22, 21, -30, 1.2, .6, True)
    return c
def s_dawn(lt, t, E, fr):
    c = A['dawn'].view(1.02 + .03 * lt / 7, .3, 0, .8)
    g, a = feather_sprite()
    for k, x in enumerate([160, 200]):
        y = -40 + lt * 55 + 6 * k; M = cv2.getRotationMatrix2D((7, 20), 25 * math.sin(t * .9 + k * 2), 1.4)
        M[0, 2] += x + 14 * math.sin(t * .8 + k) - 7; M[1, 2] += y - 20
        aw = cv2.warpAffine(a, M, (W, H)); gw = cv2.warpAffine(g, M, (W, H)); c[:] = c * (1 - aw) + gw * aw
    return c

SHOTS = [
    (0, s_hook, 0), (3.5, s_hall, 0), (7, s_young_close, 0), (10, s_elder_halo, 0), (12.5, s_wheels_close, .1),
    (16, s_eyefield, .1), (19.5, s_turn_eye, .15), (23, s_kneel, .15), (26.5, s_step, .15),
    (30, lambda lt, t, E, fr: s_eyefield(lt, t, E, fr, 1., 0.), .2), (33.5, s_tear, .25), (36, s_strike, .3),
    (40, s_fall_away, .45), (44, s_edge, .5), (48, s_dive, .55), (52, s_fall_storm, .75), (56, s_columns, .8),
    (60, s_reach_far, .85), (64, s_vortex, .85), (68, lambda lt, t, E, fr: s_void(lt, t, E, fr, False), .95),
    (72, s_void, 1.), (77, lambda lt, t, E, fr: s_void(lt, t, E, fr, False, True), 1.), (82, s_last_eye, 1.),
    (88, s_rise_light, .7), (92, lambda lt, t, E, fr: s_rise_light(lt, t, E, fr, .5), .5),
    (96, lambda lt, t, E, fr: s_rise_light(lt, t, E, fr, .85), .3), (100.6, s_meet, 0), (102.5, s_embrace, 0),
    (106, s_rose, 0), (110, s_into, 0), (114, s_white, 0), (118, s_dawn, .1)]
CAPS = [(.4, 3.3, 'he was always|one step behind'), (17, 20.5, 'they turned away'), (56.5, 59.8, 'no one asked him|to follow'),
        (77.5, 81.5, 'vers toi'), (119.5, 125, 'he asked to share|the same tomb')]
CUTS = [s[0] for s in SHOTS[1:]]

# ---------- кадр ----------
FONT = ImageFont.truetype('/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf', 11)
def caption_mask(t, fr):
    m = Image.new('L', (OW, OH), 0); d = ImageDraw.Draw(m)
    for a, b, s in CAPS:
        if a <= t < b and rnd(fr, 5) > .1:
            for k, l in enumerate(s.split('|')):
                w = d.textlength(l, font=FONT); ox = (rnd(fr, 9 + k) - .5) * 1.4
                d.text(((OW - w) / 2 + ox + 1, OH * .17 + k * 15 + 1), l, font=FONT, fill=1)
                d.text(((OW - w) / 2 + ox, OH * .17 + k * 15), l, font=FONT, fill=2)
    return np.asarray(m)

def frame(t, E, F):
    fr = int(round(t * FPS)); i = 0
    while i < len(SHOTS) - 1 and t >= SHOTS[i + 1][0]: i += 1
    t0, fn, cold = SHOTS[i]; lt = t - t0
    c = np.nan_to_num(fn(lt, t, E, fr).astype(np.float32), nan=0.)
    bloom(c, .35 + .25 * E)
    if any(t0 == x and lt < .17 for x in CUTS): c += .35
    tear = any(abs(t - x) < .1 for x in CUTS) or rnd(fr, 31) < .02
    s = cv2.resize(np.clip(c, 0, 1.5), (OW, OH), interpolation=cv2.INTER_AREA)
    if tear:
        for y in range(0, OH, 4):
            if rnd(y >> 2, fr) < .35: s[y:y + 4] = np.roll(s[y:y + 4], int((rnd(y >> 3, fr + 3) - .5) * 24), 1)
    v = np.clip(s, 0, None) ** .9 + (h2(np.arange(OW * OH).reshape(OH, OW), np.full((OH, OW), fr * 7 + 1)) - .5) * .12
    nx, ny = (np.arange(OW) - 90) / 90, (np.arange(OH)[:, None] - 160) / 160; v *= 1 - .28 * (nx * nx + ny * ny) * .5
    if rnd(fr, 44) < .3: v[:, int(rnd(fr, 45) * OW)] += .3
    lv = np.clip(np.floor(v * 4 + BAYT), 0, 4).astype(int)
    m = caption_mask(t, fr); lv[m == 1] = 1; lv[m == 2] = 4
    pal = GOLD * (1 - cold) + NIGHT * cold
    return pal[lv].astype(np.uint8)

if __name__ == '__main__':
    assets, out = sys.argv[1], sys.argv[2]
    import json; pick = json.load(open(f'{assets}/pick.json'))
    load(assets, pick)
    E = np.load(f'{assets}/vt_E.npy'); F = np.load(f'{assets}/vt_F.npy')
    em = lambda t: float(E[min(len(E) - 1, int(t * FPS))])
    if '--frames' in sys.argv:
        ts = [float(x) for x in sys.argv[sys.argv.index('--frames') + 1].split(',')]
        Image.fromarray(np.hstack([frame(t, em(t), 0) for t in ts])).save(out)
    else:
        audio = sys.argv[3]; n = int(125.2 * FPS)
        p = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{OW}x{OH}', '-r', '12', '-i', '-', '-i', audio,
                              '-vf', 'scale=1080:1920:flags=neighbor,fps=30', '-c:v', 'libx264', '-crf', '28', '-preset', 'slow', '-tune', 'grain',
                              '-x264-params', 'keyint=60', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '192k', '-shortest', '-movflags', '+faststart', out], stdin=subprocess.PIPE)
        for f in range(n):
            p.stdin.write(frame(f / FPS, em(f / FPS), 0).tobytes())
            if f % 120 == 0: print('frame', f, flush=True)
        p.stdin.close(); p.wait()
