"""Y2K-демо в серебре, полностью 2D (numpy + PIL), под припев Daidaidai.

Сценка «рабочий стол 2000-го»: голографический CD, хромовые 2D-надписи с проблеском,
каскад всплывающих окон в бит, жидкий хром (2D-метаболы), хромовые звёзды, финальный логотип.
Хром считается «по-честному» для 2D: из маски строится рельеф (distance transform),
по нормалям берётся цвет из полосатой карты отражения — классический приём Y2K-леттеринга.

python3 y2k2d.py <клип Daidaidai.mp4> out.mp4
"""
import math
import os
import subprocess
import sys
import tempfile
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy.ndimage import distance_transform_edt, gaussian_filter

import importlib.util
HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("tl", os.path.join(HERE, "timeline.py"))
TL = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(TL)

W, H, FPS = 1280, 720, TL.FPS
B = TL.B
FJP = os.path.join(HERE, "..", "daikirai_demo", "fonts", "dela-gothic-one-japanese-400-normal.ttf")
FLAT = "/usr/share/fonts/opentype/inter/InterDisplay-Black.otf"
if not os.path.exists(FLAT):
    FLAT = "/usr/share/fonts/opentype/inter/InterDisplay-Bold.otf"
FUI = "/usr/share/fonts/opentype/inter/InterDisplay-Bold.otf"
FMONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"

S1 = B[3]                 # удар бочки: CD
S2 = B[10] + 0.93         # окна
S3 = 9.28                 # жидкий хром
S4 = 12.97                # логотип Y2K
YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)


def hexc(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)], np.float32)


def ease(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def ramp(u, stops):
    """Многоцветный градиент: stops = [(pos, '#hex'), ...]."""
    pos = np.array([p for p, _ in stops], np.float32)
    cols = np.stack([hexc(c) for _, c in stops])
    u = np.clip(u, 0, 1)
    return np.stack([np.interp(u, pos, cols[:, k]) for k in range(3)], -1)


CHROME = [(0.0, "#2a2e3b"), (0.18, "#9aa1b6"), (0.36, "#ffffff"), (0.44, "#d7dbe7"), (0.47, "#4a4f60"),
          (0.55, "#20232d"), (0.66, "#8b92a8"), (0.82, "#eef0f7"), (1.0, "#b4bacb")]


# ---------------- хром из маски ----------------

def chrome_from_mask(m, bevel=10.0, tilt=0.0, sheen=0.0, tint=0.035, smooth=1.2):
    """m — float маска (h×w). Возвращает (rgb, alpha)."""
    a = np.clip(m, 0, 1)
    inside = a > 0.5
    d = distance_transform_edt(inside).astype(np.float32)
    hgt = np.sqrt(np.clip(d / bevel, 0, 1))
    hgt = gaussian_filter(hgt, smooth)
    gy, gx = np.gradient(hgt)
    h, w = m.shape
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    u = yy * 0.85 + gy * 7.5 + gx * 1.5 * tilt + 0.05 + sheen * 0.15
    rgb = ramp(u, CHROME)
    # блик на верхних скосах, затемнение внутренней кромки
    rim = np.clip(-gy * 9 - gx * 3, 0, 1)[..., None]
    rgb = rgb + rim * 0.5
    edge = np.clip(1 - d / 2.0, 0, 1)[..., None] * inside[..., None]
    rgb = rgb * (1 - 0.55 * edge)
    # голографический налёт
    ph = (yy * 3 + np.linspace(0, 1, w, dtype=np.float32)[None, :] * 2 + sheen * 2) * math.pi
    holo = np.stack([0.5 + 0.5 * np.sin(ph), 0.5 + 0.5 * np.sin(ph + 2.1), 0.5 + 0.5 * np.sin(ph + 4.2)], -1)
    rgb = rgb * (1 - tint) + holo * tint * (0.5 + rgb.mean(-1, keepdims=True))
    return np.clip(rgb, 0, 1.3), a


def text_mask(s, px, font=FJP, track=0):
    f = ImageFont.truetype(font, px)
    widths = [f.getlength(ch) for ch in s]
    wd = int(sum(widths) + track * (len(s) - 1)) + px
    im = Image.new("L", (wd, int(px * 1.5)), 0)
    d = ImageDraw.Draw(im)
    x = px // 2
    for ch, cw in zip(s, widths):
        d.text((x, int(px * 0.15)), ch, font=f, fill=255)
        x += cw + track
    bb = im.getbbox()
    im = im.crop((bb[0] - 14, bb[1] - 14, bb[2] + 14, bb[3] + 14))
    return np.asarray(im, np.float32) / 255


def outline(m, wpx):
    d = distance_transform_edt(m < 0.5)
    return np.clip(wpx + 0.5 - d, 0, 1).astype(np.float32)


class Frame:
    def __init__(self):
        self.a = np.zeros((H, W, 3), np.float32)

    def paste(self, rgb, alpha, cx, cy, scale=1.0, rot=0.0, opacity=1.0):
        """Положить RGB+alpha с центром (cx, cy), масштабом и поворотом."""
        h, w = alpha.shape
        rgba = np.dstack([np.clip(rgb, 0, 1), alpha]).astype(np.float32)
        im = Image.fromarray((rgba * 255).astype(np.uint8), "RGBA")
        if scale != 1.0:
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BICUBIC)
        if rot:
            im = im.rotate(rot, resample=Image.BICUBIC, expand=True)
        arr = np.asarray(im, np.float32) / 255
        hh, ww = arr.shape[:2]
        x0, y0 = int(cx - ww / 2), int(cy - hh / 2)
        xs0, ys0, xs1, ys1 = max(0, x0), max(0, y0), min(W, x0 + ww), min(H, y0 + hh)
        if xs1 <= xs0 or ys1 <= ys0:
            return
        sub = arr[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0]
        al = sub[..., 3:4] * opacity
        reg = self.a[ys0:ys1, xs0:xs1]
        reg[:] = reg * (1 - al) + sub[..., :3] * al

    def chrome_text(self, s, px, cx, cy, font=FJP, scale=1.0, rot=0.0, sheen=0.0, opacity=1.0, stroke=(0.12, 0.13, 0.2),
                    glow=0.0):
        m = text_mask(s, px, font)
        ol = outline(m, max(3, px // 22))
        rgb, al = chrome_from_mask(m, bevel=px / 9, sheen=sheen)
        # тёмный контур + белая тонкая кайма снаружи (наклейка)
        halo = outline(m, max(6, px // 11))
        base = np.zeros(m.shape + (3,), np.float32) + 1.0
        base = base * (halo - ol)[..., None] + np.array(stroke) * ol[..., None]
        a_all = np.clip(halo, 0, 1)
        rgb_all = base * (1 - al[..., None]) + rgb * al[..., None]
        if glow:
            g = gaussian_filter(a_all, px / 6) * glow
            self.paste(np.ones_like(rgb_all), np.clip(g, 0, 1), cx, cy, scale, rot, opacity * 0.8)
        self.paste(rgb_all, a_all, cx, cy, scale, rot, opacity)
        # проблеск: диагональная светлая полоса бежит по буквам
        if 0 < sheen < 1:
            hh, ww = m.shape
            yy_, xx_ = np.mgrid[0:hh, 0:ww].astype(np.float32)
            band = np.exp(-((xx_ + yy_ * 0.6 - sheen * (ww + hh)) / (px * 0.25)) ** 2)
            self.paste(np.ones((hh, ww, 3), np.float32), band * al * 0.85, cx, cy, scale, rot, opacity)

    def add(self, layer, k=1.0):
        self.a = self.a + layer * k


# ---------------- элементы ----------------

def bg_silver(t, hue=0.0, dark=0.0):
    u = YY / H
    base = ramp(u * 0.9 + 0.05, [(0, "#c9cedb"), (0.45, "#f3f4f9"), (0.5, "#dfe2ec"), (0.75, "#b9bfd0"), (1, "#9aa1b8")])
    ph = (XX / W * 1.6 - YY / H * 0.8 + t * 0.12 + hue) * math.pi
    holo = np.stack([0.5 + 0.5 * np.sin(ph), 0.5 + 0.5 * np.sin(ph + 2.1), 0.5 + 0.5 * np.sin(ph + 4.2)], -1)
    g = base.mean(-1, keepdims=True)
    a = (base * 0.6 + g * 0.4) * 0.93 + holo * 0.07
    # мягкие «пузырьковые» круги-блики
    for k in range(5):
        cx = W * (0.15 + 0.18 * k) + 30 * math.sin(t * 0.5 + k)
        cy = H * (0.3 + 0.12 * math.sin(k * 1.7 + t * 0.4))
        r = 90 + 25 * k
        d = np.hypot(XX - cx, YY - cy)
        a = a + (np.clip(1 - np.abs(d - r) / 2.2, 0, 1) * 0.10)[..., None]
    return a * (1 - dark)


def star_mask(size, r1=1.0, r2=0.16, n=4, rot=0.0):
    s = int(size * 2 + 24)
    im = Image.new("L", (s, s), 0)
    d = ImageDraw.Draw(im)
    pts = []
    for k in range(n * 2):
        a = rot + math.pi / 2 + k * math.pi / n
        r = (r1 if k % 2 == 0 else r2) * size
        # вогнутые лучи: промежуточные точки по дуге
        pts.append((s / 2 + r * math.cos(a), s / 2 - r * math.sin(a)))
    d.polygon(pts, fill=255)
    return np.asarray(im.filter(ImageFilter.GaussianBlur(0.6)), np.float32) / 255


def chrome_star(fr, cx, cy, size, rot=0.0, opacity=1.0):
    m = star_mask(size, rot=rot)
    rgb, al = chrome_from_mask(m, bevel=size / 5, tint=0.1)
    ol = outline(m, 2)
    rgb = rgb * (1 - ol[..., None] * (1 - al[..., None])) + 0
    fr.paste(rgb, np.maximum(al, ol * 0.6), cx, cy, opacity=opacity)


def cd(fr, cx, cy, R, ang, t, opacity=1.0):
    """Голографический диск: радужные секторы вращаются, серебряная основа, отверстие."""
    x0, y0 = int(cx - R - 4), int(cy - R - 4)
    sz = int(2 * R + 8)
    yy, xx = np.mgrid[0:sz, 0:sz].astype(np.float32)
    dx, dy = xx - sz / 2, yy - sz / 2
    r = np.hypot(dx, dy)
    th = np.arctan2(dy, dx) - ang
    al = np.clip(R - r, 0, 1) * np.clip(r - R * 0.16, 0, 1)
    # серебро: блик-«бабочка» (две светлые четверти)
    sil = 0.62 + 0.33 * np.cos(2 * th) ** 8 + 0.06 * np.sin(r * 0.9)
    ph = th * 2 + r / R * 7 + t * 0.6
    holo = np.stack([0.5 + 0.5 * np.sin(ph), 0.5 + 0.5 * np.sin(ph + 2.1), 0.5 + 0.5 * np.sin(ph + 4.2)], -1)
    wedge = (np.abs(np.sin(th * 2)) ** 3)[..., None]
    rgb = sil[..., None] * (1 - 0.55 * wedge) + holo * 0.55 * wedge
    # дорожки и прозрачная середина
    tracks = (0.04 * np.sin(r * 2.3))[..., None]
    rgb = rgb + tracks
    hub = (r < R * 0.32)[..., None]
    rgb = np.where(hub, np.array([0.86, 0.88, 0.93]) - 0.08 * (r / (R * 0.32))[..., None], rgb)
    ring = np.clip(1 - np.abs(r - R * 0.32) / 1.5, 0, 1) + np.clip(1 - np.abs(r - R) / 1.5, 0, 1) * 0.8 + \
        np.clip(1 - np.abs(r - R * 0.16) / 1.5, 0, 1)
    rgb = rgb * (1 - 0.6 * ring[..., None])
    fr.paste(rgb, al, cx, cy, opacity=opacity)


def popup(d, box, title, body, t_since, icon="!"):
    """Окно в духе Win2000/Mac OS 9, серебряная рамка."""
    if t_since < 0:
        return
    x0, y0, x1, y1 = box
    k = min(1.0, t_since / 0.09)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    sx = 0.6 + 0.4 * k
    x0, x1 = cx - (cx - x0) * sx, cx + (x1 - cx) * sx
    y0, y1 = cy - (cy - y0) * sx, cy + (y1 - cy) * sx
    d.rounded_rectangle((x0 + 6, y0 + 8, x1 + 6, y1 + 8), 10, fill=(40, 44, 60, 70))
    d.rounded_rectangle((x0, y0, x1, y1), 10, fill=(232, 235, 243, 255), outline=(96, 102, 122, 255), width=2)
    for i in range(int(x1 - x0 - 8)):
        c = int(165 + 70 * i / max(1, (x1 - x0)))
        d.line([(x0 + 4 + i, y0 + 4), (x0 + 4 + i, y0 + 28)], fill=(c - 70, c - 45, 255, 255))
    d.line([(x0 + 4, y0 + 5), (x1 - 4, y0 + 5)], fill=(255, 255, 255, 200))
    d.text((x0 + 14, y0 + 5), title, font=ImageFont.truetype(FUI, 18), fill=(255, 255, 255))
    for j, c in enumerate([(255, 95, 90), (255, 200, 80), (110, 210, 120)]):
        d.ellipse((x1 - 66 + j * 19, y0 + 9, x1 - 54 + j * 19, y0 + 21), fill=c, outline=(80, 80, 95))
    d.ellipse((x0 + 18, y0 + 44, x0 + 58, y0 + 84), fill=(255, 214, 80), outline=(120, 90, 20), width=2)
    d.text((x0 + 33, y0 + 48), icon, font=ImageFont.truetype(FUI, 26), fill=(60, 40, 0))
    d.text((x0 + 74, y0 + 48), body, font=ImageFont.truetype(FJP, 27), fill=(40, 44, 60))
    for j, lab in enumerate(["OK", "キャンセル"]):
        bx = x1 - 210 + j * 100
        d.rounded_rectangle((bx, y1 - 42, bx + 88, y1 - 14), 8, fill=(246, 247, 251), outline=(110, 116, 135), width=2)
        d.text((bx + 44, y1 - 28), lab, font=ImageFont.truetype(FJP, 14), fill=(40, 44, 60), anchor="mm")


def cursor(d, x, y, s=1.0):
    pts = [(0, 0), (0, 26), (7, 20), (12, 31), (16, 29), (11, 18), (20, 18)]
    d.polygon([(x + px * s, y + py * s) for px, py in pts], fill=(255, 255, 255), outline=(20, 20, 30))


def metaballs(t, n=6, burst=0.0, zoom=1.0):
    """Жидкий хром: поле метаболов в половинном разрешении, хром по градиенту поля."""
    h, w = H // 2, W // 2
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    F = np.zeros((h, w), np.float32)
    for k in range(n):
        a = t * (0.7 + 0.13 * k) + k * 1.1
        rr = (95 + 45 * math.sin(t * 1.1 + k * 0.8)) * (1 + burst * (1.5 + 0.3 * k)) * zoom
        cx = w / 2 + rr * math.cos(a)
        cy = h / 2 + rr * 0.55 * math.sin(a * 1.3)
        r = (26 + 6 * math.sin(t * 2 + k)) * zoom * (1 - 0.3 * burst)
        F += r * r / ((xx - cx) ** 2 + (yy - cy) ** 2 + 1)
    m = np.clip((F - 1.0) * 40, 0, 1)
    rgb, _ = chrome_from_mask(m, bevel=22 * zoom, tint=0.05, smooth=2.6)
    ol = outline(m, 1.5)
    rgb = rgb * (1 - 0.7 * ol[..., None] * (m < 0.5)[..., None])
    m = np.maximum(m, ol * 0.9)
    im = Image.fromarray((np.dstack([np.clip(rgb, 0, 1), m]) * 255).astype(np.uint8), "RGBA").resize((W, H), Image.BICUBIC)
    arr = np.asarray(im, np.float32) / 255
    return arr[..., :3], arr[..., 3]


def bloom(a, thr=0.86, amount=0.5):
    s = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).resize((W // 4, H // 4), Image.BILINEAR)
    s = np.clip(np.asarray(s, np.float32) / 255 - thr, 0, 1) / (1 - thr)
    b = Image.fromarray((s * 255).astype(np.uint8))
    b1 = np.asarray(b.filter(ImageFilter.GaussianBlur(3)).resize((W, H), Image.BILINEAR), np.float32) / 255
    b2 = np.asarray(b.filter(ImageFilter.GaussianBlur(12)).resize((W, H), Image.BILINEAR), np.float32) / 255
    return a + (b1 * 0.5 + b2 * 0.7) * amount


def glints(img_rgba, a, t, n=6):
    lum = a.mean(axis=2)
    small = lum[::8, ::8]
    idx = np.argsort(small.ravel())[::-1][:n * 8]
    pts = []
    for i in idx:
        y, x = divmod(int(i), small.shape[1])
        y, x = y * 8 + 4, x * 8 + 4
        if lum[min(y, H - 1), min(x, W - 1)] < 0.97:
            continue
        if all((x - px) ** 2 + (y - py) ** 2 > 110 ** 2 for px, py in pts):
            pts.append((x, y))
        if len(pts) >= n:
            break
    d = ImageDraw.Draw(img_rgba, "RGBA")
    for k, (x, y) in enumerate(pts):
        L = 16 + 10 * ((k + int(t * 6)) % 3) + 12 * TL.pulse(t)
        for ang, ln, wd in ((0, L, 2), (math.pi / 2, L, 2), (math.pi / 4, L * 0.35, 1), (-math.pi / 4, L * 0.35, 1)):
            dx, dy = math.cos(ang) * ln, math.sin(ang) * ln
            d.line([(x - dx, y - dy), (x + dx, y + dy)], fill=(255, 255, 255, 230), width=wd)


def sparkles(img, t, n=14, seed=5):
    d = ImageDraw.Draw(img, "RGBA")
    r = np.random.default_rng(seed)
    for k in range(n):
        x, y = r.uniform(40, W - 40), r.uniform(40, H - 40)
        ph = r.uniform(0, 6.28)
        s = (5 + 12 * max(0.0, math.sin(t * 2.6 + ph))) * (1 + 0.6 * TL.pulse(t))
        if s < 7:
            continue
        al = int(230 * min(1, (s - 6) / 8))
        for ang in (0, math.pi / 2):
            dx, dy = math.cos(ang) * s, math.sin(ang) * s
            d.line([(x - dx, y - dy), (x + dx, y + dy)], fill=(255, 255, 255, al), width=2)


# ---------------- кадр ----------------

def render(fi):
    t = TL.T0 + fi / FPS
    p = TL.pulse(t)
    fr = Frame()
    slow = BREAK = TL.BREAK0 <= t < TL.DROP
    zoom = 1 + 0.025 * p

    if t < S1:                                              # интро: хромовая звезда растёт, загрузка
        k = (t - TL.T0) / (S1 - TL.T0)
        fr.a = bg_silver(t, dark=0.08)
        chrome_star(fr, W / 2, H / 2 - 30, 60 + 150 * ease(k), rot=t * 1.5)
        for j in range(4):
            a = t * 1.3 + j * math.pi / 2
            chrome_star(fr, W / 2 + 260 * math.cos(a) * ease(k), H / 2 - 30 + 150 * math.sin(a) * ease(k), 22, rot=-t * 3)
    elif t < S2:                                            # CD + «ダイ»×5 + «ダイキライ»
        fr.a = bg_silver(t)
        spin = (t - S1) * 7.0
        R = 230 * zoom
        cd(fr, W / 2, H / 2, R, spin, t)
        for j in range(6):
            a = -t * 0.9 + j * math.pi / 3
            chrome_star(fr, W / 2 + 320 * math.cos(a), H / 2 + 230 * math.sin(a), 20 + 8 * p, rot=t * 2 + j)
        pos = [(250, 170, -10), (1030, 170, 10), (230, 540, 8), (1050, 540, -8), (640, 110, 3)]
        for k, dt in enumerate(TL.DAI):
            if dt <= t < TL.KIRAI + 0.08:
                age = t - dt
                x, y, rr = pos[k]
                fr.chrome_text("ダイ", 120, x, y, scale=1 + 0.6 * math.exp(-age * 14), rot=rr, sheen=min(0.99, age * 2.5))
        if t >= TL.KIRAI:
            age = t - TL.KIRAI
            fr.chrome_text("ダイキライ", 170, W / 2, H / 2, scale=1 + 0.5 * math.exp(-age * 10), rot=-3,
                           sheen=min(0.99, age * 1.6), glow=0.6)
    elif t < S3:                                            # каскад окон
        fr.a = bg_silver(t, hue=1.0)
        cd(fr, 1050, 560, 130, (t - S1) * 4.0, t, opacity=0.9)
        img = Image.fromarray((np.clip(fr.a, 0, 1) * 255).astype(np.uint8)).convert("RGBA")
        d = ImageDraw.Draw(img, "RGBA")
        beats = [b for b in B if S2 - 0.05 <= b < S3]
        texts = ["ダイキライ.exe が応答しません", "アナタのコト.txt を削除しますか？", "最早バイバイしたい", "OMG ……",
                 "予測不能なエラー", "解 が見つかりません", "ダイダイダイダイ", "メモリ不足：ダイスキ", "再起動しますか？"]
        for k, b in enumerate(beats):
            x0 = 120 + (k * 63) % 420
            y0 = 70 + (k * 41) % 300
            popup(d, (x0, y0, x0 + 640, y0 + 210), f"Error {k + 1:02d}", texts[k % len(texts)], t - b)
        cursor(d, 760 + 60 * math.sin(t * 2), 380 + 40 * math.sin(t * 2.7), 1.4)
        fr.a = np.asarray(img.convert("RGB"), np.float32) / 255
    elif t < S4:                                            # жидкий хром
        fr.a = bg_silver(t, hue=2.0, dark=0.05)
        rgb, al = metaballs(t - S3, zoom=1.0 + 0.04 * p)
        fr.a = fr.a * (1 - al[..., None]) + rgb * al[..., None]
        for j in range(5):
            a = t * 0.8 + j * 1.25
            chrome_star(fr, W / 2 + 430 * math.cos(a), H / 2 + 250 * math.sin(a), 26, rot=-t * 2)
    elif t < TL.BREAK0:                                     # хромовый логотип
        fr.a = bg_silver(t, hue=3.0)
        cd(fr, W / 2, H / 2, 300, (t - S1) * 3, t, opacity=0.35)
        age = t - S4
        fr.chrome_text("Y2K", 300, W / 2, H / 2 - 20, font=FLAT, scale=(1 + 0.3 * math.exp(-age * 8)) * zoom, rot=-6,
                       sheen=(age * 0.8) % 1.0, glow=0.5)
        fr.chrome_text("SILVER  MILLENNIUM", 54, W / 2, H / 2 + 175, font=FUI, opacity=ease(age * 3))
        for j, (x, y) in enumerate([(230, 160), (1060, 140), (200, 560), (1090, 580)]):
            chrome_star(fr, x, y, 34 + 14 * p, rot=t * 2 * (-1) ** j)
    elif t < TL.DROP:                                       # брейк: медленный крупный жидкий хром
        k = (t - TL.BREAK0) / (TL.DROP - TL.BREAK0)
        fr.a = bg_silver(t, hue=2.0, dark=0.12 * (1 - k))
        rgb, al = metaballs(S4 - S3 + (t - TL.BREAK0) * 0.35, zoom=1.6 + 0.7 * k)
        fr.a = fr.a * (1 - al[..., None]) + rgb * al[..., None]
    else:                                                   # дроп: взрыв капель, финал
        age = t - TL.DROP
        fr.a = bg_silver(t)
        rgb, al = metaballs(S4 - S3 + 0.6 + age, burst=ease(age / 0.6), zoom=1.3)
        fr.a = fr.a * (1 - al[..., None]) + rgb * al[..., None]
        cd(fr, W / 2, H / 2, 200 + 40 * ease(age / 0.5), age * 9, t, opacity=min(1, age * 3))
        fr.chrome_text("ダイキライ", 150, W / 2, H / 2, scale=1 + 0.4 * math.exp(-age * 8), rot=-2,
                       sheen=min(0.99, age * 1.2), glow=0.7)
        for j in range(8):
            a = j * math.pi / 4 + age * 0.8
            rr = 120 + 420 * ease(age / 0.8)
            chrome_star(fr, W / 2 + rr * math.cos(a), H / 2 + rr * 0.6 * math.sin(a), 30, rot=age * 4 + j)

    a = bloom(fr.a, amount=0.55)
    fl = 0.0
    for tt, amp, dec in TL.FLASHES:
        if t >= tt:
            fl = max(fl, amp * math.exp(-(t - tt) * dec))
    if t < S1:
        fl = max(fl, 0.6 * max(0, (t - (S1 - 0.3)) / 0.3) ** 2)
    a = a * (1 - fl) + fl
    out = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).convert("RGBA")
    glints(out, a, t)
    sparkles(out, t)
    if t > TL.T1 - 0.7:
        k = min(1.0, (t - (TL.T1 - 0.7)) / 0.7)
        out = Image.blend(out, Image.new("RGBA", (W, H), (240, 242, 248, 255)), k)
    return out.convert("RGB")


def work(fi):
    p = os.path.join(TMP, f"y{fi:05d}.jpg")
    if not os.path.exists(p):
        render(fi).save(p, quality=94)
    return fi


TMP = os.environ.get("FRAMES_DIR") or tempfile.mkdtemp(prefix="y2k2d_")

if __name__ == "__main__":
    song, out = sys.argv[1], sys.argv[2]
    os.makedirs(TMP, exist_ok=True)
    n = int((TL.T1 - TL.T0) * FPS)
    with Pool(os.cpu_count()) as pool:
        list(pool.imap_unordered(work, range(n)))
    dur = n / FPS
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", os.path.join(TMP, "y%05d.jpg"),
                    "-ss", str(43.0 + TL.T0), "-t", str(dur), "-i", song, "-map", "0:v", "-map", "1:a",
                    "-af", f"afade=t=in:d=0.2,afade=t=out:st={dur - 0.8}:d=0.8",
                    "-c:v", "libx264", "-crf", "21", "-preset", "slow", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out], check=True)
    print("done", out)
