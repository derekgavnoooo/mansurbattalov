"""«Ручной» рендер: бумага, дрожащие линии с нажимом, акварельные заливки, рукописный текст."""
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import os

W, H = 1280, 720
SS = 2
RW, RH = W * SS, H * SS
HERE = os.path.dirname(os.path.abspath(__file__))

INK = np.array([0.17, 0.14, 0.19])
PENCIL_BLUE = np.array([0.45, 0.62, 0.86])


def hexc(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])


# ---------------- геометрия ----------------

def catmull(pts, closed=False, step=2.0):
    P = np.asarray(pts, float)
    if len(P) < 2:
        return P
    if closed:
        P = np.vstack([P[-1], P, P[0], P[1]])
    else:
        P = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        m = max(2, int(np.linalg.norm(p2 - p1) / step))
        t = np.linspace(0, 1, m, endpoint=False)[:, None]
        out.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(P[1][None] if closed else P[-2][None])
    return np.vstack(out)


def path(*pieces):
    """Склейка сплайнов: на стыках остаются острые углы (кончики прядей)."""
    out = []
    for pc in pieces:
        c = catmull(pc)
        out.append(c if not out else c[1:])
    return np.vstack(out)


def ellipse(cx, cy, rx, ry, rot=0.0, n=48):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x, y = rx * np.cos(a), ry * np.sin(a)
    c, s = math.cos(rot), math.sin(rot)
    return np.stack([cx + c * x - s * y, cy + s * x + c * y], 1)


def resample(P, step):
    P = np.asarray(P, float)
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    if d[-1] < 1e-6:
        return P[:1].repeat(2, 0), 0.0
    n = max(2, int(d[-1] / step) + 1)
    s = np.linspace(0, d[-1], n)
    return np.stack([np.interp(s, d, P[:, 0]), np.interp(s, d, P[:, 1])], 1), d[-1]


def resample_n(P, n):
    P = np.asarray(P, float)
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    s = np.linspace(0, d[-1], n)
    return np.stack([np.interp(s, d, P[:, 0]), np.interp(s, d, P[:, 1])], 1)


def plen(P):
    return float(np.linalg.norm(np.diff(np.asarray(P, float), axis=0), axis=1).sum())


class Xf:
    """Аффинное преобразование: поворот вокруг pivot + сдвиг + масштаб."""

    def __init__(self, tx=0, ty=0, rot=0.0, pivot=(0, 0), sx=1.0, sy=1.0, parent=None):
        self.tx, self.ty, self.rot, self.pivot, self.sx, self.sy, self.parent = tx, ty, rot, pivot, sx, sy, parent

    def __call__(self, P):
        P = np.asarray(P, float).copy()
        px, py = self.pivot
        P[:, 0] = (P[:, 0] - px) * self.sx
        P[:, 1] = (P[:, 1] - py) * self.sy
        c, s = math.cos(self.rot), math.sin(self.rot)
        P = np.stack([c * P[:, 0] - s * P[:, 1] + px + self.tx, s * P[:, 0] + c * P[:, 1] + py + self.ty], 1)
        return self.parent(P) if self.parent else P


# ---------------- шумы ----------------

def value_noise(h, w, cell, rng):
    gh, gw = h // cell + 3, w // cell + 3
    g = Image.fromarray((rng.random((gh, gw)) * 255).astype(np.uint8))
    g = g.resize((gw * cell, gh * cell), Image.BICUBIC)
    return np.asarray(g, np.float32)[:h, :w] / 255.0


class Textures:
    def __init__(self, seed=1):
        rng = np.random.default_rng(seed)
        PW, PH = RW + 256, RH + 256
        self.pad = 128
        g = rng.standard_normal((PH, PW)).astype(np.float32)
        g = np.asarray(Image.fromarray(((g * 40) + 128).clip(0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.7)), np.float32) / 255
        self.grain = (g - g.mean()) / (g.std() + 1e-6)          # зерно бумаги
        self.pigment = (value_noise(PH, PW, 90, rng) * 0.55 + value_noise(PH, PW, 26, rng) * 0.3
                        + value_noise(PH, PW, 9, rng) * 0.15)   # неравномерность краски
        self.reveal = value_noise(PH, PW, 60, rng) * 0.7 + value_noise(PH, PW, 14, rng) * 0.3
        self.paper = self._paper(rng)

    def _paper(self, rng):
        base = np.ones((RH, RW, 3), np.float32) * np.array([0.958, 0.943, 0.905], np.float32)
        lf = value_noise(RH, RW, 300, rng) - 0.5
        mf = value_noise(RH, RW, 40, rng) - 0.5
        g = self.grain[:RH, :RW]
        shade = 1 + lf * 0.035 + mf * 0.012 + g * 0.012
        img = base * shade[..., None]
        # волокна бумаги
        fib = Image.new("L", (RW, RH), 0)
        d = ImageDraw.Draw(fib)
        for _ in range(900):
            x, y = rng.random() * RW, rng.random() * RH
            a = rng.random() * np.pi
            L = rng.uniform(6, 30)
            d.line([(x, y), (x + math.cos(a) * L, y + math.sin(a) * L)], fill=int(rng.uniform(20, 60)), width=1)
        fib = np.asarray(fib.filter(ImageFilter.GaussianBlur(0.6)), np.float32) / 255
        img *= (1 - fib * 0.06)[..., None]
        # виньетка (как при съёмке листа на телефон)
        yy, xx = np.mgrid[0:RH, 0:RW].astype(np.float32)
        r = np.sqrt(((xx - RW * 0.48) / RW) ** 2 + ((yy - RH * 0.45) / RH) ** 2)
        img *= (1 - 0.13 * np.clip(r - 0.25, 0, 1) ** 1.4)[..., None]
        # след от стёртого карандаша и пятно графита у ребра ладони
        sm = Image.new("L", (RW, RH), 0)
        d = ImageDraw.Draw(sm)
        d.ellipse([RW * 0.80, RH * 0.82, RW * 1.02, RH * 1.05], fill=60)
        d.ellipse([RW * 0.05, RH * 0.88, RW * 0.2, RH * 0.98], fill=30)
        sm = np.asarray(sm.filter(ImageFilter.GaussianBlur(60)), np.float32) / 255
        img *= (1 - sm * 0.09 * (0.7 + 0.6 * self.pigment[:RH, :RW]))[..., None]
        return img.clip(0, 1)

    def crop(self, arr, x0, y0, x1, y1, ox=0, oy=0):
        p = self.pad
        return arr[y0 + p + oy:y1 + p + oy, x0 + p + ox:x1 + p + ox]


# ---------------- холст ----------------

class Canvas:
    def __init__(self, tex, seed, boil=1.0):
        self.tex = tex
        self.img = tex.paper.copy()
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        r = self.rng
        # поле «дрожания» линий для этого рисунка (line boil)
        self.waves = [(r.uniform(0, 2 * np.pi), 2 * np.pi / r.uniform(70, 260), r.uniform(0, 2 * np.pi),
                       r.uniform(0.25, 0.6) * boil) for _ in range(6)]
        # несовпадение заливки с контуром (каждый кадр раскрашен «заново»)
        self.color_shift = r.normal(0, 1.0, 2) + np.array([1.4, 1.0])

    def field(self, P):
        dx = np.zeros(len(P))
        dy = np.zeros(len(P))
        for i, (ang, k, ph, a) in enumerate(self.waves):
            proj = P[:, 0] * math.cos(ang) + P[:, 1] * math.sin(ang)
            v = a * np.sin(proj * k + ph)
            if i % 2:
                dx += v
            else:
                dy += v
        return P + np.stack([dx, dy], 1)

    def _bbox(self, P, pad):
        x0 = max(0, int(P[:, 0].min() - pad))
        y0 = max(0, int(P[:, 1].min() - pad))
        x1 = min(RW, int(P[:, 0].max() + pad) + 1)
        y1 = min(RH, int(P[:, 1].max() + pad) + 1)
        return x0, y0, x1, y1

    def _poly_mask(self, polys, box, blur=0.0):
        x0, y0, x1, y1 = box
        m = Image.new("L", (x1 - x0, y1 - y0), 0)
        d = ImageDraw.Draw(m)
        for P, val in polys:
            d.polygon([(float(x - x0), float(y - y0)) for x, y in P], fill=val)
        if blur:
            m = m.filter(ImageFilter.GaussianBlur(blur))
        return np.asarray(m, np.float32) / 255

    def occlude(self, polys, blur=0.9):
        """Непрозрачный слой: закрывает всё, что «за» ним, бумагой."""
        Ps = [self.field(np.asarray(p, float)) * SS for p in polys]
        allp = np.vstack(Ps)
        box = self._bbox(allp, 6)
        if box[2] <= box[0] or box[3] <= box[1]:
            return
        a = self._poly_mask([(p, 255) for p in Ps], box, blur * SS)[..., None]
        x0, y0, x1, y1 = box
        self.img[y0:y1, x0:x1] = self.img[y0:y1, x0:x1] * (1 - a) + self.tex.paper[y0:y1, x0:x1] * a

    def fill(self, poly, color, alpha=0.85, mode="mul", wobble=1.0, edge=0.45, tex=0.35,
             clip=None, holes=(), reveal=1.0, shift=True, blur=1.1, key=0):
        r = np.random.default_rng(self.seed * 1000 + key)
        P = np.asarray(poly, float)
        # дрожь края заливки — сильнее, чем у линии
        P, _ = resample(P, 4.0)
        P = self.field(P)
        P += r.normal(0, 0.6 * wobble, P.shape)
        if shift:
            P += self.color_shift * wobble
        P *= SS
        box = self._bbox(P, 14 * SS)
        if box[2] <= box[0] or box[3] <= box[1]:
            return
        x0, y0, x1, y1 = box
        polys = [(P, 255)] + [((self.field(np.asarray(h, float)) * SS), 0) for h in holes]
        a = self._poly_mask(polys, box, blur * SS)
        if clip is not None:
            cl = [(self.field(np.asarray(c, float)) * SS, 255) for c in clip]
            a = a * self._poly_mask(cl, box, 0.6 * SS)
        if a.max() <= 0:
            return
        ox, oy = r.integers(-100, 100, 2)
        pig = self.tex.crop(self.tex.pigment, x0, y0, x1, y1, ox, oy)
        if edge:
            b = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(5 * SS)), np.float32) / 255
            rim = np.clip((a - b) * 2.4, 0, 1)
            a = a * (1 + edge * rim)
        dens = a * (1 - tex + tex * (0.35 + 1.1 * pig)) * alpha
        if reveal < 1:
            rv = self.tex.crop(self.tex.reveal, x0, y0, x1, y1, ox // 3, oy // 3)
            yy = np.linspace(y0, y1, y1 - y0)[:, None] / RH
            thr = rv * 0.3 + yy * 0.7
            dens = dens * np.clip((reveal * 1.25 - thr) / 0.12, 0, 1)
        dens = np.clip(dens, 0, 1)[..., None]
        reg = self.img[y0:y1, x0:x1]
        col = np.asarray(color, np.float32)
        if mode == "mul":
            reg *= 1 - dens * (1 - col)
        else:
            reg[:] = reg * (1 - dens) + col * dens

    def stroke(self, pts, w=2.0, color=INK, alpha=0.92, taper=(0.18, 0.28), prog=1.0,
               grain=0.3, sketch=0.25, overshoot=2.0, key=0, smooth_press=True, closed=False):
        if prog <= 0:
            return 0.0
        r = np.random.default_rng(self.seed * 7919 + key)
        P = np.asarray(pts, float)
        if closed:
            P = np.vstack([P, P[:1]])
        P, L = resample(P, 1.0)
        if L < 0.5:
            return 0.0
        if overshoot and not closed and len(P) > 4:
            for end in (0, -1):
                nb = 1 if end == 0 else -2
                t = P[end] - P[nb]
                t /= np.linalg.norm(t) + 1e-9
                ext = r.uniform(0, overshoot)
                n = int(ext)
                if n:
                    add = P[end] + t * np.arange(1, n + 1)[:, None]
                    P = np.vstack([add[::-1], P]) if end == 0 else np.vstack([P, add])
        # каждый штрих кладётся рукой отдельно: свой сдвиг и поворот
        c = P.mean(0)
        ang = r.normal(0, 0.006)
        cs, sn = math.cos(ang), math.sin(ang)
        Q = P - c
        P = np.stack([cs * Q[:, 0] - sn * Q[:, 1], sn * Q[:, 0] + cs * Q[:, 1]], 1) + c + r.normal(0, 0.45, 2)
        P = self.field(P)
        n = len(P)
        if prog < 1:
            P = P[:max(2, int(n * prog))]
        s = np.linspace(0, 1, n)[:len(P)]
        ta, tb = taper
        press = np.minimum(1, s / max(ta, 1e-3)) ** 0.55 * np.minimum(1, (1 - s) / max(tb, 1e-3)) ** 0.55
        press = 0.18 + 0.82 * press if not closed else np.ones_like(s)
        k = np.convolve(r.standard_normal(len(s) + 40), np.hanning(31) / np.hanning(31).sum(), "same")[20:20 + len(s)]
        press *= 0.88 + 0.5 * k
        rad = np.clip(w * press * 0.5, 0.25, None) * SS
        Ps = P * SS
        box = self._bbox(Ps, rad.max() + 4)
        if box[2] <= box[0] or box[3] <= box[1]:
            return L
        x0, y0, x1, y1 = box
        m = Image.new("L", (x1 - x0, y1 - y0), 0)
        d = ImageDraw.Draw(m)
        for (x, y), rr in zip(Ps, rad):
            d.ellipse([x - x0 - rr, y - y0 - rr, x - x0 + rr, y - y0 + rr], fill=255)
        m = m.filter(ImageFilter.GaussianBlur(0.45 * SS))
        a = np.asarray(m, np.float32) / 255
        g = self.tex.crop(self.tex.grain, x0, y0, x1, y1)
        a = np.clip(a * (1 + grain * 0.5 * g) - grain * 0.15 * (g < -0.6), 0, 1) * alpha
        reg = self.img[y0:y1, x0:x1]
        reg *= 1 - a[..., None] * (1 - np.asarray(color, np.float32))
        if sketch and r.random() < sketch and prog >= 1 and L > 25:
            # вторая, неуверенная проводка рядом
            off = r.normal(0, 1.1, 2)
            seg0 = r.uniform(0, 0.35)
            seg1 = r.uniform(0.55, 1)
            Q = np.asarray(pts, float)
            Q, _ = resample(Q, 1.0)
            Q = Q[int(len(Q) * seg0):max(int(len(Q) * seg0) + 3, int(len(Q) * seg1))] + off
            self.stroke(Q, w * 0.55, color, alpha * 0.4, taper=(0.4, 0.4), grain=grain + 0.2,
                        sketch=0, overshoot=0, key=key + 991)
        return L

    def hatch(self, poly, angle, gap, w=0.9, color=INK, alpha=0.5, key=0, length_jit=0.25, prog=1.0):
        """Штриховка внутри полигона."""
        r = np.random.default_rng(self.seed * 31 + key)
        P = np.asarray(poly, float)
        c, s = math.cos(angle), math.sin(angle)
        # координаты в повёрнутой системе
        u = P[:, 0] * c + P[:, 1] * s
        v = -P[:, 0] * s + P[:, 1] * c
        lines = []
        vv = v.min() + gap * r.uniform(0.2, 0.8)
        while vv < v.max():
            xs = []
            for i in range(len(P)):
                v1, v2 = v[i], v[(i + 1) % len(P)]
                if (v1 - vv) * (v2 - vv) < 0:
                    t = (vv - v1) / (v2 - v1)
                    xs.append(u[i] + t * (u[(i + 1) % len(P)] - u[i]))
            xs.sort()
            for a, b in zip(xs[::2], xs[1::2]):
                L = b - a
                a += r.uniform(0, length_jit) * L * 0.5
                b -= r.uniform(0, length_jit) * L * 0.5
                if b - a > 3:
                    vj = vv + r.normal(0, gap * 0.12)
                    pa = np.array([a * c - vj * s, a * s + vj * c])
                    pb = np.array([b * c - (vj + r.normal(0, 1)) * s, b * s + (vj + r.normal(0, 1)) * c])
                    lines.append((pa, pb))
            vv += gap * r.uniform(0.8, 1.2)
        n = int(len(lines) * prog)
        for i, (pa, pb) in enumerate(lines[:n]):
            self.stroke([pa, (pa + pb) / 2 + r.normal(0, 0.5, 2), pb], w, color, alpha * r.uniform(0.7, 1.1),
                        taper=(0.15, 0.5), sketch=0, overshoot=1, key=key * 100 + i, grain=0.6)

    def text(self, s, x, y, size, color=INK, alpha=0.9, prog=1.0, key=0, rot=0.0, colors=None, spacing=0.0):
        """Рукописный текст: каждая буква своя (наклон, базовая линия, размер)."""
        fonts = Fonts.get(int(size * SS))
        rs = np.random.default_rng(key)            # «почерк» — постоянный
        rb = np.random.default_rng(self.seed * 13 + key)  # дрожь — от кадра к кадру
        cx = x
        n_show = int(round(len(s) * prog)) if prog < 1 else len(s)
        ca, sa = math.cos(rot), math.sin(rot)
        for i, ch in enumerate(s):
            f = fonts[1] if ("Ѐ" <= ch <= "ӿ") else fonts[0]
            adv = f.getlength(ch) / SS
            base_rot = rs.normal(0, 3.5)
            base_dy = rs.normal(0, size * 0.035)
            base_sc = 1 + rs.normal(0, 0.025)
            if i < n_show and ch != " ":
                col = colors[i] if colors is not None else color
                px = x + (cx - x) * ca
                py = y + (cx - x) * sa + base_dy
                pp = self.field(np.array([[px, py]]))[0] + rb.normal(0, 0.35, 2)
                ff = Fonts.get(int(size * SS * base_sc))
                ff = ff[1] if ("Ѐ" <= ch <= "ӿ") else ff[0]
                bb = ff.getbbox(ch)
                gw, gh = bb[2] + 8, bb[3] + 8
                im = Image.new("L", (gw, gh), 0)
                ImageDraw.Draw(im).text((4, 4), ch, font=ff, fill=255)
                im = im.rotate(base_rot + rb.normal(0, 0.8) - math.degrees(rot), resample=Image.BICUBIC, expand=True)
                im = im.filter(ImageFilter.GaussianBlur(0.4 * SS))
                a = np.asarray(im, np.float32) / 255
                X0 = int(pp[0] * SS)
                Y0 = int(pp[1] * SS - size * SS * 0.75)
                x0, y0 = max(0, X0), max(0, Y0)
                x1, y1 = min(RW, X0 + a.shape[1]), min(RH, Y0 + a.shape[0])
                if x1 > x0 and y1 > y0:
                    a = a[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0]
                    g = self.tex.crop(self.tex.grain, x0, y0, x1, y1)
                    a = np.clip(a * (1 + 0.15 * g), 0, 1) * alpha
                    self.img[y0:y1, x0:x1] *= 1 - a[..., None] * (1 - np.asarray(col, np.float32))
            cx += adv + spacing + rs.normal(0, size * 0.015)
        return cx

    def tape(self, cx, cy, w, h, ang, key=0):
        """Кусок малярного скотча."""
        r = np.random.default_rng(key)
        top = [(-w / 2 + i * w / 8, -h / 2 + r.normal(0, 0.8)) for i in range(9)]
        right = [(w / 2 + r.normal(0, 2.2), -h / 2 + i * h / 6) for i in range(1, 6)]
        bot = [(w / 2 - i * w / 8, h / 2 + r.normal(0, 0.8)) for i in range(9)]
        left = [(-w / 2 + r.normal(0, 2.2), h / 2 - i * h / 6) for i in range(1, 6)]
        P = np.array(top + right + bot + left)
        c, s = math.cos(ang), math.sin(ang)
        P = np.stack([c * P[:, 0] - s * P[:, 1] + cx, s * P[:, 0] + c * P[:, 1] + cy], 1) * SS
        box = self._bbox(P, 8)
        x0, y0, x1, y1 = box
        a = self._poly_mask([(P, 255)], box, 0.8 * SS)
        pig = self.tex.crop(self.tex.pigment, x0, y0, x1, y1, 40, 40)
        dens = (a * (0.62 + 0.2 * pig))[..., None]
        reg = self.img[y0:y1, x0:x1]
        reg[:] = reg * (1 - dens) + np.array([0.93, 0.88, 0.72]) * dens
        # тень от скотча
        reg *= 1 - (np.clip(a - np.roll(a, -4, 0), 0, 1) * 0.08)[..., None]

    def result(self):
        im = Image.fromarray((np.clip(self.img, 0, 1) * 255).astype(np.uint8))
        return im.resize((W, H), Image.LANCZOS)


class Fonts:
    _c = {}

    @classmethod
    def get(cls, px):
        if px not in cls._c:
            cls._c[px] = (ImageFont.truetype(os.path.join(HERE, "fonts", "caveat-latin.ttf"), px),
                          ImageFont.truetype(os.path.join(HERE, "fonts", "caveat-cyrillic.ttf"), px))
        return cls._c[px]
