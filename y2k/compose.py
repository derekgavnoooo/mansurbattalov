"""Пост-обработка Y2K-демо: апскейл 3D-кадров, свечение, блики-звёзды, хроматика,
интерфейс нулевых (окно, загрузка, курсор, «sparkle»), вспышки в бит, сборка с треком.

python3 compose.py <папка_рендеров> <клип Daidaidai.mp4> out.mp4
"""
import math
import os
import subprocess
import sys
import tempfile
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import importlib.util
HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("tl", os.path.join(HERE, "timeline.py"))
TL = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(TL)

W, H, FPS = 1280, 720, TL.FPS
FONT = "/usr/share/fonts/opentype/inter/InterDisplay-Bold.otf"
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
SRC = None


def f(path, px):
    return ImageFont.truetype(path, px)


def bloom(a, thr=0.80, amount=0.55):
    s = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).resize((W // 4, H // 4), Image.BILINEAR)
    s = np.clip(np.asarray(s, np.float32) / 255 - thr, 0, 1) / (1 - thr)
    b = Image.fromarray((s * 255).astype(np.uint8))
    b1 = np.asarray(b.filter(ImageFilter.GaussianBlur(4)).resize((W, H), Image.BILINEAR), np.float32) / 255
    b2 = np.asarray(b.filter(ImageFilter.GaussianBlur(14)).resize((W, H), Image.BILINEAR), np.float32) / 255
    return a + (b1 * 0.6 + b2 * 0.6) * amount


def star_glints(a, n=7, strength=1.0):
    """Четырёхлучевые блики на самых ярких точках хрома (фирменная деталь Y2K)."""
    lum = a.mean(axis=2)
    small = lum[::8, ::8]
    idx = np.argsort(small.ravel())[::-1][:n * 6]
    pts = []
    for i in idx:
        y, x = divmod(int(i), small.shape[1])
        y, x = y * 8 + 4, x * 8 + 4
        if lum[min(y, H - 1), min(x, W - 1)] < 0.92:
            continue
        if all((x - px) ** 2 + (y - py) ** 2 > 90 ** 2 for px, py in pts):
            pts.append((x, y))
        if len(pts) >= n:
            break
    lay = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(lay)
    for k, (x, y) in enumerate(pts):
        L = (26 + 14 * (k % 3)) * strength
        for ang, ln in ((0, L), (math.pi / 2, L), (math.pi / 4, L * 0.35), (-math.pi / 4, L * 0.35)):
            dx, dy = math.cos(ang) * ln, math.sin(ang) * ln
            d.line([(x - dx, y - dy), (x + dx, y + dy)], fill=255, width=2)
    lay = lay.filter(ImageFilter.GaussianBlur(1.3))
    g = np.asarray(lay, np.float32)[..., None] / 255
    return a + g * 0.9


def chroma(a, k):
    k = int(round(k))
    if k < 1:
        return a
    out = a.copy()
    out[..., 0] = np.roll(a[..., 0], k, axis=1)
    out[..., 2] = np.roll(a[..., 2], -k, axis=1)
    return out


def bevel_box(d, box, fill=(232, 235, 243), light=(255, 255, 255), dark=(120, 126, 145), r=10):
    x0, y0, x1, y1 = box
    d.rounded_rectangle(box, r, fill=fill, outline=dark, width=2)
    d.line([(x0 + r, y0 + 2), (x1 - r, y0 + 2)], fill=light, width=2)


def ui(img, t):
    """Окно в духе ОС/плееров начала 2000-х: заголовок-градиент, кнопки, прогресс, VU."""
    d = ImageDraw.Draw(img, "RGBA")
    if t < TL.B[3]:
        # загрузка
        k = min(1.0, (t - TL.T0) / (TL.B[3] - TL.T0 - 0.1))
        x0, y0, x1, y1 = 440, 560, 840, 640
        bevel_box(d, (x0, y0, x1, y1), fill=(226, 229, 238, 235))
        d.text((x0 + 16, y0 + 10), "LOADING  silver_world.exe", font=f(MONO, 15), fill=(60, 64, 80))
        d.rounded_rectangle((x0 + 16, y0 + 38, x1 - 16, y0 + 62), 6, fill=(170, 176, 192), outline=(110, 116, 135))
        n = int(22 * k)
        for i in range(n):
            bx = x0 + 20 + i * 16.4
            d.rounded_rectangle((bx, y0 + 42, bx + 12, y0 + 58), 3, fill=(120, 170, 255) if i % 2 else (160, 205, 255))
        d.text((x1 - 70, y0 + 10), f"{int(k * 100):3d}%", font=f(MONO, 15), fill=(60, 64, 80))
        return
    # плеер в углу
    x0, y0, x1, y1 = 34, 590, 420, 676
    bevel_box(d, (x0, y0, x1, y1), fill=(226, 229, 238, 220))
    for i in range(0, x1 - x0 - 4, 2):                        # заголовок-градиент
        c = int(150 + 80 * i / (x1 - x0))
        d.line([(x0 + 2 + i, y0 + 3), (x0 + 2 + i, y0 + 22)], fill=(c - 60, c - 30, 255, 230))
    d.text((x0 + 10, y0 + 4), "♫ daidaidai.mp3  —  SilverAmp", font=f(FONT, 14), fill=(255, 255, 255))
    for k, c in enumerate([(255, 95, 90), (255, 200, 80), (110, 210, 120)]):
        d.ellipse((x1 - 62 + k * 18, y0 + 6, x1 - 50 + k * 18, y0 + 18), fill=c, outline=(90, 90, 100))
    # VU-полоски по огибающей вокала
    env = TL.env_at(t)
    for i in range(18):
        h = int(4 + 34 * max(0.0, min(1.0, env * (0.6 + 0.6 * math.sin(i * 1.7 + t * 9)) + TL.pulse(t) * 0.5)))
        bx = x0 + 12 + i * 11
        d.rectangle((bx, y1 - 10 - h, bx + 7, y1 - 10), fill=(110 + 6 * i, 150 + 4 * i, 255))
    prog = (t - TL.T0) / (TL.T1 - TL.T0)
    d.rounded_rectangle((x0 + 220, y1 - 30, x1 - 14, y1 - 18), 5, fill=(170, 176, 192), outline=(110, 116, 135))
    d.rounded_rectangle((x0 + 222, y1 - 28, x0 + 222 + (x1 - x0 - 238) * prog, y1 - 20), 4, fill=(140, 190, 255))
    m, s = divmod(int(43 + t), 60)
    d.text((x0 + 222, y0 + 30), f"{m}:{s:02d}", font=f(MONO, 18), fill=(60, 64, 80))
    # курсор-стрелка, плавает
    cx, cy = 1120 + 40 * math.sin(t * 0.7), 120 + 25 * math.sin(t * 1.1)
    arrow = [(cx, cy), (cx, cy + 26), (cx + 7, cy + 20), (cx + 12, cy + 31), (cx + 16, cy + 29), (cx + 11, cy + 18), (cx + 20, cy + 18)]
    d.polygon(arrow, fill=(255, 255, 255), outline=(30, 30, 40))


def sparkles(img, t, n=10):
    d = ImageDraw.Draw(img, "RGBA")
    r = np.random.default_rng(5)
    for k in range(n):
        x, y = r.uniform(60, W - 60), r.uniform(60, H - 160)
        ph = r.uniform(0, 6.28)
        s = (6 + 10 * max(0.0, math.sin(t * 3.0 + ph))) * (1 + 0.6 * TL.pulse(t))
        if s < 7:
            continue
        a = int(220 * min(1, (s - 6) / 8))
        for ang in (0, math.pi / 2):
            dx, dy = math.cos(ang) * s, math.sin(ang) * s
            d.line([(x - dx, y - dy), (x + dx, y + dy)], fill=(255, 255, 255, a), width=2)
        d.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(255, 255, 255, a))


def render(fi):
    t = TL.T0 + fi / FPS
    src = os.path.join(SRC, f"r{fi:05d}.png")
    im = Image.open(src).convert("RGB").resize((W, H), Image.LANCZOS)
    im = im.filter(ImageFilter.UnsharpMask(radius=1.2, percent=60, threshold=2))
    a = np.asarray(im, np.float32) / 255
    a = bloom(a, amount=0.7 if t >= TL.DROP else 0.5)
    a = star_glints(a, strength=1.0 + 0.6 * TL.pulse(t))
    # переливы: лёгкий розово-голубой градиент поверх (голографическая плёнка)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    ph = (xx / W * 2.0 + yy / H * 1.2 + t * 0.25) * math.pi
    holo = np.stack([0.5 + 0.5 * np.sin(ph), 0.5 + 0.5 * np.sin(ph + 2.1), 0.5 + 0.5 * np.sin(ph + 4.2)], 2)
    a = a * (1 - 0.07) + holo * 0.07 * a.mean(axis=2, keepdims=True) * 1.6
    # удары
    fl = 0.0
    for tt, amp, dec in TL.FLASHES:
        if t >= tt:
            fl = max(fl, amp * math.exp(-(t - tt) * dec))
    a = a * (1 - fl) + fl
    ca = 1.2 + 6 * max(TL.pulse(t), math.exp(-max(0, t - TL.KIRAI) * 7) if t >= TL.KIRAI else 0,
                       math.exp(-max(0, t - TL.DROP) * 5) if t >= TL.DROP else 0)
    a = chroma(a, ca)
    v = 1 - 0.28 * np.clip(np.hypot((xx - W / 2) / W, (yy - H / 2) / H) - 0.28, 0, 1) ** 1.4
    a = a * v[..., None]
    a = a + np.random.default_rng(fi).normal(0, 0.012, (H, W, 1))
    out = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).convert("RGBA")
    sparkles(out, t)
    ui(out, t)
    # скан-линии (очень легко), затемнение в конце
    sl = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ds = ImageDraw.Draw(sl)
    for y in range(0, H, 3):
        ds.line([(0, y), (W, y)], fill=(0, 0, 0, 14))
    out.alpha_composite(sl)
    if t > TL.T1 - 0.7:
        k = min(1.0, (t - (TL.T1 - 0.7)) / 0.7)
        out = Image.blend(out, Image.new("RGBA", (W, H), (235, 238, 245, 255)), k)
    return out.convert("RGB")


def work(fi):
    p = os.path.join(TMP, f"c{fi:05d}.jpg")
    if not os.path.exists(p):
        render(fi).save(p, quality=94)
    return fi


TMP = os.environ.get("COMP_DIR") or tempfile.mkdtemp(prefix="y2kc_")

if __name__ == "__main__":
    SRC, song, out = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(TMP, exist_ok=True)
    n = len([x for x in os.listdir(SRC) if x.startswith("r") and x.endswith(".png")])
    with Pool(os.cpu_count(), initializer=globals().__setitem__, initargs=("SRC", SRC)) as pool:
        list(pool.imap_unordered(work, range(n)))
    dur = n / FPS
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", os.path.join(TMP, "c%05d.jpg"),
                    "-ss", str(43.0 + TL.T0), "-t", str(dur), "-i", song, "-map", "0:v", "-map", "1:a",
                    "-af", f"afade=t=in:d=0.2,afade=t=out:st={dur - 0.8}:d=0.8",
                    "-c:v", "libx264", "-crf", "21", "-preset", "slow", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out], check=True)
    print("done", out)
