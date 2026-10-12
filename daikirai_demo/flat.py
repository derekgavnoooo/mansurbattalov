"""Плоский «цифровой» рендер в духе MV Amala: чистые заливки, чёрный контур с живой
дрожью, halftone, текстура ткани, глитч, хроматическая аберрация."""
import json
import math
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 1280, 720
SS = 2
RW, RH = W * SS, H * SS
HERE = os.path.dirname(os.path.abspath(__file__))
LINE = np.array([0.09, 0.06, 0.11])


def hexc(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])


# ---------------- геометрия ----------------

def catmull(pts, closed=False, step=2.0):
    P = np.asarray(pts, float)
    if len(P) < 2:
        return P
    P = np.vstack([P[-1], P, P[0], P[1]]) if closed else np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
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


def offset_curve(P, d):
    P = np.asarray(P, float)
    t = np.gradient(P, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
    n = np.stack([-t[:, 1], t[:, 0]], 1)
    return P + n * (np.asarray(d)[:, None] if np.ndim(d) else d)


def capsule(a, b, ra, rb, n=14):
    """Трубка от a до b (рука, палец)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    L = np.linalg.norm(d) + 1e-9
    ang = math.atan2(d[1], d[0])
    pts = []
    for k in range(n + 1):
        t = ang + math.pi / 2 + math.pi * k / n
        pts.append(b + rb * np.array([math.cos(t - math.pi), math.sin(t - math.pi)]))
    for k in range(n + 1):
        t = ang - math.pi / 2 + math.pi * k / n
        pts.append(a + ra * np.array([math.cos(t + math.pi), math.sin(t + math.pi)]))
    return np.array(pts)


class Xf:
    def __init__(self, tx=0, ty=0, rot=0.0, pivot=(0, 0), sx=1.0, sy=1.0, parent=None):
        self.tx, self.ty, self.rot, self.pivot, self.sx, self.sy, self.parent = tx, ty, rot, pivot, sx, sy, parent

    def __call__(self, P):
        P = np.asarray(P, float).reshape(-1, 2).copy()
        px, py = self.pivot
        P[:, 0] = (P[:, 0] - px) * self.sx
        P[:, 1] = (P[:, 1] - py) * self.sy
        c, s = math.cos(self.rot), math.sin(self.rot)
        P = np.stack([c * P[:, 0] - s * P[:, 1] + px + self.tx, s * P[:, 0] + c * P[:, 1] + py + self.ty], 1)
        return self.parent(P) if self.parent else P

    def scale(self):
        s = abs(self.sx)
        return s * (self.parent.scale() if self.parent else 1.0)


def value_noise(h, w, cell, rng):
    gh, gw = h // cell + 3, w // cell + 3
    g = Image.fromarray((rng.random((gh, gw)) * 255).astype(np.uint8)).resize((gw * cell, gh * cell), Image.BICUBIC)
    return np.asarray(g, np.float32)[:h, :w] / 255.0


# ---------------- шрифты ----------------

class Fonts:
    _maps = {}
    _cache = {}

    @classmethod
    def font_for(cls, fam, ch, px):
        if fam not in cls._maps:
            m = json.load(open(os.path.join(HERE, "fonts", fam + ".json")))
            cls._maps[fam] = {c: f for f, cs in m.items() for c in cs}
        f = cls._maps[fam].get(ch)
        if f is None:
            return None
        k = (f, px)
        if k not in cls._cache:
            cls._cache[k] = ImageFont.truetype(os.path.join(HERE, "fonts", f), px)
        return cls._cache[k]


# ---------------- текстуры ----------------

class Tex:
    def __init__(self, seed=3):
        rng = np.random.default_rng(seed)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        # текстура ткани/холста, как в Daidai
        weave = (np.sin(xx * 1.9) * np.sin(yy * 1.9) * 0.5 + np.sin((xx + yy) * 0.7) * 0.2)
        n = value_noise(H, W, 3, rng) - 0.5
        lf = value_noise(H, W, 90, rng) - 0.5
        self.canvas = 1 + 0.035 * weave + 0.05 * n + 0.04 * lf
        r = np.sqrt(((xx - W / 2) / W) ** 2 + ((yy - H / 2) / H) ** 2)
        self.vignette = 1 - 0.55 * np.clip(r - 0.28, 0, 1) ** 1.6
        self.grain = [rng.standard_normal((H, W)).astype(np.float32) for _ in range(6)]


# ---------------- холст ----------------

class Canvas:
    def __init__(self, seed, boil=1.0, bg=(0, 0, 0)):
        self.img = np.ones((RH, RW, 3), np.float32) * np.asarray(bg, np.float32)
        self.seed = seed
        r = np.random.default_rng(seed)
        self.waves = [(r.uniform(0, 2 * np.pi), 2 * np.pi / r.uniform(80, 280), r.uniform(0, 2 * np.pi),
                       r.uniform(0.2, 0.45) * boil) for _ in range(6)]

    def field(self, P):
        P = np.asarray(P, float)
        dx = np.zeros(len(P))
        dy = np.zeros(len(P))
        for i, (ang, k, ph, a) in enumerate(self.waves):
            v = a * np.sin((P[:, 0] * math.cos(ang) + P[:, 1] * math.sin(ang)) * k + ph)
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

    def mask(self, polys, box, blur=0.5):
        x0, y0, x1, y1 = box
        m = Image.new("L", (x1 - x0, y1 - y0), 0)
        d = ImageDraw.Draw(m)
        for P, val in polys:
            if len(P) >= 3:
                d.polygon([(float(x - x0), float(y - y0)) for x, y in P], fill=val)
        if blur:
            m = m.filter(ImageFilter.GaussianBlur(blur))
        return np.asarray(m, np.float32) / 255

    def fill(self, poly, color, alpha=1.0, clip=None, holes=(), jitter=True, pattern=None):
        P = np.asarray(poly, float)
        if len(P) < 3:
            return
        P = (self.field(P) if jitter else P) * SS
        box = self._bbox(P, 4)
        if box[2] <= box[0] or box[3] <= box[1]:
            return
        x0, y0, x1, y1 = box
        a = self.mask([(P, 255)] + [((self.field(np.asarray(h, float)) * SS), 0) for h in holes], box, 0.6)
        if clip is not None:
            a = a * self.mask([(self.field(np.asarray(c, float)) * SS, 255) for c in clip], box, 0.6)
        if pattern is not None:
            a = a * pattern(x0, y0, x1, y1)
        a = (a * alpha)[..., None]
        reg = self.img[y0:y1, x0:x1]
        reg[:] = reg * (1 - a) + np.asarray(color, np.float32) * a

    def rect(self, x0, y0, x1, y1, color, alpha=1.0):
        X0, Y0, X1, Y1 = [int(round(v * SS)) for v in (x0, y0, x1, y1)]
        X0, Y0, X1, Y1 = max(0, X0), max(0, Y0), min(RW, X1), min(RH, Y1)
        if X1 > X0 and Y1 > Y0:
            reg = self.img[Y0:Y1, X0:X1]
            reg[:] = reg * (1 - alpha) + np.asarray(color, np.float32) * alpha

    def gradient(self, c_top, c_bot, y0=0, y1=H):
        t = np.clip((np.arange(RH) / SS - y0) / max(1, y1 - y0), 0, 1)[:, None, None]
        self.img[:] = np.asarray(c_top, np.float32) * (1 - t) + np.asarray(c_bot, np.float32) * t

    def stroke(self, pts, w=3.0, color=LINE, alpha=1.0, taper=(0.12, 0.15), closed=False, key=0,
               overshoot=1.2, jitter=True):
        r = np.random.default_rng(self.seed * 7919 + key)
        P = np.asarray(pts, float)
        if closed:
            P = np.vstack([P, P[:1], P[1:2]])
        P, L = resample(P, 0.9)
        if L < 0.5:
            return
        if overshoot and not closed and len(P) > 4:
            for end in (0, -1):
                nb = 1 if end == 0 else -2
                t = P[end] - P[nb]
                t /= np.linalg.norm(t) + 1e-9
                n = int(r.uniform(0, overshoot) / 0.9)
                if n:
                    add = P[end] + t * 0.9 * np.arange(1, n + 1)[:, None]
                    P = np.vstack([add[::-1], P]) if end == 0 else np.vstack([P, add])
        if jitter:
            P = self.field(P + r.normal(0, 0.25, 2))
        n = len(P)
        s = np.linspace(0, 1, n)
        if closed:
            press = np.ones(n)
        else:
            ta, tb = taper
            press = np.minimum(1, s / max(ta, 1e-3)) ** 0.5 * np.minimum(1, (1 - s) / max(tb, 1e-3)) ** 0.5
            press = 0.25 + 0.75 * press
        k = np.convolve(r.standard_normal(n + 60), np.hanning(41) / np.hanning(41).sum(), "same")[30:30 + n]
        press = press * (0.92 + 0.35 * k)
        rad = np.clip(w * press * 0.5, 0.3, None) * SS
        Ps = P * SS
        box = self._bbox(Ps, rad.max() + 3)
        if box[2] <= box[0] or box[3] <= box[1]:
            return
        x0, y0, x1, y1 = box
        m = Image.new("L", (x1 - x0, y1 - y0), 0)
        d = ImageDraw.Draw(m)
        for (x, y), rr in zip(Ps, rad):
            d.ellipse([x - x0 - rr, y - y0 - rr, x - x0 + rr, y - y0 + rr], fill=255)
        a = (np.asarray(m.filter(ImageFilter.GaussianBlur(0.5)), np.float32) / 255 * alpha)[..., None]
        reg = self.img[y0:y1, x0:x1]
        reg[:] = reg * (1 - a) + np.asarray(color, np.float32) * a

    def union(self, polys, color, w=3.0, line=LINE, key=0):
        """Залитое объединение фигур с общим контуром (через расширение маски)."""
        Ps = [self.field(np.asarray(p, float)) * SS for p in polys]
        allp = np.vstack(Ps)
        pad = int(w * SS) + 4
        box = self._bbox(allp, pad)
        if box[2] <= box[0] or box[3] <= box[1]:
            return
        x0, y0, x1, y1 = box
        m = Image.new("L", (x1 - x0, y1 - y0), 0)
        d = ImageDraw.Draw(m)
        for P in Ps:
            d.polygon([(float(x - x0), float(y - y0)) for x, y in P], fill=255)
        k = max(3, int(round(w * SS)) | 1)
        o = m.filter(ImageFilter.MaxFilter(k)).filter(ImageFilter.GaussianBlur(0.6))
        m = m.filter(ImageFilter.GaussianBlur(0.6))
        reg = self.img[y0:y1, x0:x1]
        a = (np.asarray(o, np.float32) / 255)[..., None]
        reg[:] = reg * (1 - a) + np.asarray(line, np.float32) * a
        a = (np.asarray(m, np.float32) / 255)[..., None]
        reg[:] = reg * (1 - a) + np.asarray(color, np.float32) * a

    def shape(self, poly, color, w=3.0, line=LINE, key=0, **kw):
        self.fill(poly, color, **kw)
        if w:
            self.stroke(poly, w, line, closed=True, key=key)

    def halftone(self, color, spacing=9, angle=0.6, density=None, alpha=1.0, box=None, max_r=0.5):
        """Растровые точки; density(x, y) -> 0..1 в координатах кадра."""
        x0, y0, x1, y1 = box or (0, 0, W, H)
        sp = spacing * SS
        yy, xx = np.mgrid[int(y0 * SS):int(y1 * SS), int(x0 * SS):int(x1 * SS)].astype(np.float32)
        c, s = math.cos(angle), math.sin(angle)
        u = (xx * c + yy * s) / sp
        v = (-xx * s + yy * c) / sp
        du = u - np.round(u)
        dv = v - np.round(v)
        dist = np.sqrt(du ** 2 + dv ** 2)
        dens = density(xx / SS, yy / SS) if density else np.ones_like(xx)
        rr = np.sqrt(np.clip(dens, 0, 1)) * max_r
        a = np.clip((rr - dist) * sp * 0.7, 0, 1) * alpha
        reg = self.img[int(y0 * SS):int(y1 * SS), int(x0 * SS):int(x1 * SS)]
        reg[:] = reg * (1 - a[..., None]) + np.asarray(color, np.float32) * a[..., None]

    def text(self, s, x, y, px, fam="dotgothic16", color=(1, 1, 1), outline=None, ow=0, anchor="l",
             rot=0.0, alpha=1.0, spacing=0.0, jit=0.0, scale=1.0):
        """Текст с посимвольным выбором файла шрифта. Возвращает ширину."""
        px_ss = max(4, int(px * SS * scale))
        glyphs = []
        width = 0.0
        for ch in s:
            f = Fonts.font_for(fam, ch, px_ss) or Fonts.font_for("zen-maru-gothic", ch, px_ss)
            if ch in " 　":
                adv = px_ss * (0.32 if ch == " " else 1.0)
                glyphs.append((None, None, adv))
            elif f is None:
                glyphs.append((None, None, px_ss * 0.5))
                adv = px_ss * 0.5
            else:
                adv = f.getlength(ch)
                glyphs.append((ch, f, adv))
            width += adv + spacing * SS * scale
        width -= spacing * SS * scale
        ow_ss = int(round(ow * SS * scale))
        pad = ow_ss + 6
        im_w, im_h = int(width) + pad * 2, int(px_ss * 1.45) + pad * 2
        fill_m = Image.new("L", (im_w, im_h), 0)
        out_m = Image.new("L", (im_w, im_h), 0)
        df, do = ImageDraw.Draw(fill_m), ImageDraw.Draw(out_m)
        r = np.random.default_rng(self.seed * 3 + hash(s) % 1000)
        cx = pad
        for ch, f, adv in glyphs:
            if ch is not None:
                dy = r.normal(0, jit * SS) if jit else 0
                if ow_ss:
                    do.text((cx, pad + dy), ch, font=f, fill=255, stroke_width=ow_ss, stroke_fill=255)
                df.text((cx, pad + dy), ch, font=f, fill=255)
            cx += adv + spacing * SS * scale
        if rot:
            fill_m = fill_m.rotate(math.degrees(rot), resample=Image.BICUBIC, expand=True)
            out_m = out_m.rotate(math.degrees(rot), resample=Image.BICUBIC, expand=True)
        fa = np.asarray(fill_m, np.float32) / 255
        oa = np.asarray(out_m, np.float32) / 255
        hw, hh = fa.shape[1], fa.shape[0]
        ax = {"l": 0, "c": hw / 2, "r": hw}[anchor]
        X0 = int(x * SS - ax)
        Y0 = int(y * SS - hh / 2)
        xs0, ys0 = max(0, X0), max(0, Y0)
        xs1, ys1 = min(RW, X0 + hw), min(RH, Y0 + hh)
        if xs1 <= xs0 or ys1 <= ys0:
            return width / SS
        sl = (slice(ys0 - Y0, ys1 - Y0), slice(xs0 - X0, xs1 - X0))
        reg = self.img[ys0:ys1, xs0:xs1]
        if outline is not None and ow_ss:
            a = (oa[sl] * alpha)[..., None]
            reg[:] = reg * (1 - a) + np.asarray(outline, np.float32) * a
        a = (fa[sl] * alpha)[..., None]
        reg[:] = reg * (1 - a) + np.asarray(color, np.float32) * a
        return width / SS

    def ruby_line(self, segs, cx, y, px, color=(1, 1, 1), outline=LINE, ow=3, fam="dotgothic16"):
        """Строка с фуриганой над кандзи, по центру cx."""
        widths = [self.measure(t, px, fam) for t, _ in segs]
        x = cx - sum(widths) / 2
        for (t, ruby), w in zip(segs, widths):
            self.text(t, x, y, px, fam, color, outline, ow)
            if ruby:
                rp = px * 0.45
                rw = self.measure(ruby, rp, fam)
                sp = 0.0
                if rw < w and len(ruby) > 1:
                    sp = (w - rw) / (len(ruby) - 1) * 0.6
                    rw = self.measure(ruby, rp, fam) + sp * (len(ruby) - 1)
                self.text(ruby, x + w / 2 - rw / 2, y - px * 0.78, rp, fam, color, outline, max(1, ow * 0.6), spacing=sp)
            x += w

    def measure(self, s, px, fam="dotgothic16"):
        px_ss = max(4, int(px * SS))
        w = 0.0
        for ch in s:
            if ch == " ":
                w += px_ss * 0.32
            elif ch == "　":
                w += px_ss
            else:
                f = Fonts.font_for(fam, ch, px_ss) or Fonts.font_for("zen-maru-gothic", ch, px_ss)
                w += f.getlength(ch) if f else px_ss * 0.5
        return w / SS

    def finish(self, tex, fx):
        """Пост-обработка в разрешении кадра. fx: dict(ca, glitch, shake, grain_i, flash, tint)."""
        im = Image.fromarray((np.clip(self.img, 0, 1) * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS)
        a = np.asarray(im, np.float32) / 255
        a = a * tex.canvas[..., None]
        r = np.random.default_rng(fx.get("seed", 0))
        # глитч: сдвиг горизонтальных полос
        g = fx.get("glitch", 0.0)
        if g > 0:
            n = int(3 + 10 * g)
            for _ in range(n):
                y0 = int(r.uniform(0, H))
                hh = int(r.uniform(4, 10 + 50 * g))
                sh = int(r.normal(0, 40 * g))
                a[y0:y0 + hh] = np.roll(a[y0:y0 + hh], sh, axis=1)
                if r.random() < 0.4:
                    a[y0:y0 + hh, :, r.integers(0, 3)] *= r.uniform(0.3, 1.4)
        # хроматическая аберрация
        ca = fx.get("ca", 1.0) + 7 * g
        if ca >= 0.5:
            k = int(round(ca))
            a[..., 0] = np.roll(a[..., 0], k, axis=1)
            a[..., 2] = np.roll(a[..., 2], -k, axis=1)
        # тряска камеры
        sx, sy = fx.get("shake", (0, 0))
        if sx or sy:
            a = np.roll(a, (int(sy), int(sx)), axis=(0, 1))
        if fx.get("flash", 0) > 0:
            f = fx["flash"]
            a = a * (1 - f) + np.asarray(fx.get("flash_color", (1, 1, 1))) * f
        tint = fx.get("tint")
        if tint is not None:
            a = a * np.asarray(tint, np.float32)
        a = a * tex.vignette[..., None]
        a = a + tex.grain[fx.get("grain_i", 0) % len(tex.grain)][..., None] * 0.022
        # чуть приподнятые фиолетовые тени, как в цветокоре клипа
        a = a * 0.93 + np.array([0.035, 0.0, 0.055], np.float32)
        return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
