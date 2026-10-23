"""«Bad Apple в Desmos»: каждый кадр видео переводится в набор параметрических кривых
(кубические кривые Безье) и рисуется как график в стиле Desmos.

Контуры: порог по тёмным линиям → marching squares (skimage) → упрощение ломаной →
сегменты Catmull-Rom = кубические Безье, каждый сегмент — одно уравнение (x(t), y(t)).

python3 desmos_trace.py <видео> <начало_с> <длительность_с> out.mp4
"""
import math
import os
import subprocess
import sys
import tempfile
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from skimage import measure

W, H, FPS = 1280, 720, 24
SS = 2
GW, GH = 640, 360                      # сетка анализа
FONT = "/usr/share/fonts/opentype/inter/InterDisplay-Medium.otf"
FMATH = "/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf"
COLORS = [(199, 68, 64), (45, 112, 179), (56, 140, 70), (96, 66, 166), (250, 121, 33), (0, 0, 0)]
SRC, T0 = None, 0.0


def frame_at(i):
    return Image.open(os.path.join(SRC, f"s{i + 1:05d}.png")).convert("L")


def curves(gray):
    g = np.asarray(gray.resize((GW, GH), Image.BILINEAR), np.float32) / 255
    dark = (g < 0.45).astype(np.float32)
    dark = np.asarray(Image.fromarray((dark * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8)), np.float32) / 255
    segs = []
    for c in measure.find_contours(dark, 0.5):
        if len(c) < 12:
            continue
        p = measure.approximate_polygon(c, tolerance=0.9)
        if len(p) < 3:
            continue
        P = p[:, ::-1]                           # (x, y)
        n = len(P)
        closed = np.allclose(P[0], P[-1])
        for i in range(n - 1):
            p0 = P[i - 1] if i > 0 else (P[-2] if closed else P[i])
            p1, p2 = P[i], P[i + 1]
            p3 = P[i + 2] if i + 2 < n else (P[1] if closed else P[i + 1])
            c1 = p1 + (p2 - p0) / 6
            c2 = p2 - (p3 - p1) / 6
            segs.append((p1, c1, c2, p2))
    return segs


def to_screen(pt):
    return ((pt[0] / GW) * W) * SS, ((pt[1] / GH) * H) * SS


def render(fi):
    segs = curves(frame_at(fi))
    im = Image.new("RGB", (W * SS, H * SS), (255, 255, 255))
    d = ImageDraw.Draw(im)
    # сетка Desmos
    step = 40 * SS
    for k in range(0, W * SS, step // 5):
        d.line([(k, 0), (k, H * SS)], fill=(244, 244, 244) if k % step else (220, 220, 220), width=1 if k % step else 2)
    for k in range(0, H * SS, step // 5):
        d.line([(0, k), (W * SS, k)], fill=(244, 244, 244) if k % step else (220, 220, 220), width=1 if k % step else 2)
    cx, cy = W * SS // 2, H * SS // 2
    d.line([(cx, 0), (cx, H * SS)], fill=(40, 40, 40), width=3)
    d.line([(0, cy), (W * SS, cy)], fill=(40, 40, 40), width=3)
    fnum = ImageFont.truetype(FONT, 22)
    for k in range(-15, 16):
        if k:
            d.text((cx + k * step - 8, cy + 8), str(k), font=fnum, fill=(60, 60, 60))
    for k in range(-8, 9):
        if k:
            d.text((cx + 10, cy - k * step - 12), str(k), font=fnum, fill=(60, 60, 60))
    t = np.linspace(0, 1, 9)[:, None]
    for j, (p0, p1, p2, p3) in enumerate(segs):
        B = ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + 3 * (1 - t) * t * t * p2 + (t ** 3) * p3
        pts = [to_screen(q) for q in B]
        d.line(pts, fill=COLORS[(j // 40) % len(COLORS)], width=5, joint="curve")
    im = im.resize((W, H), Image.LANCZOS)
    d = ImageDraw.Draw(im, "RGBA")
    # панель уравнений, как в Desmos
    d.rectangle((0, 0, 400, H), fill=(255, 255, 255, 235))
    d.line([(400, 0), (400, H)], fill=(210, 210, 210), width=2)
    fm = ImageFont.truetype(FMATH, 15)
    fs = ImageFont.truetype(FONT, 15)
    rows = min(len(segs), 13)
    for r in range(rows):
        p0, p1, p2, p3 = [q / np.array([GW, GH]) * np.array([30, -18]) + np.array([-15, 9]) for q in segs[r * max(1, len(segs) // 13)]]
        y = 14 + r * 52
        col = COLORS[r % len(COLORS)]
        d.rectangle((0, y - 8, 400, y + 42), outline=(232, 232, 232))
        d.text((8, y + 6), str(r + 1), font=fs, fill=(150, 150, 150))
        d.ellipse((30, y + 8, 44, y + 22), fill=col)
        eq = f"((1-t)³·{p0[0]:.2f}+3t(1-t)²·{p1[0]:.2f}+3t²(1-t)·{p2[0]:.2f}+t³·{p3[0]:.2f},"
        eq2 = f" (1-t)³·{p0[1]:.2f}+3t(1-t)²·{p1[1]:.2f}+3t²(1-t)·{p2[1]:.2f}+t³·{p3[1]:.2f})"
        d.text((54, y), eq, font=fm, fill=(30, 30, 30))
        d.text((54, y + 18), eq2, font=fm, fill=(30, 30, 30))
    d.rectangle((0, H - 46, 400, H), fill=(245, 245, 245, 255))
    d.text((14, H - 34), f"уравнений в кадре: {len(segs)}     0 ≤ t ≤ 1", font=ImageFont.truetype(FONT, 19), fill=(40, 40, 40))
    # маленький оригинал для сравнения
    th = Image.open(os.path.join(SRC, f"s{fi + 1:05d}.png")).convert("RGB").resize((256, 144))
    im.paste(th, (W - 272, H - 160))
    d.rectangle((W - 273, H - 161, W - 15, H - 15), outline=(120, 120, 120), width=2)
    return im


def work(fi):
    p = os.path.join(TMP, f"d{fi:05d}.png")
    if not os.path.exists(p):
        render(fi).save(p)
    return fi


TMP = os.environ.get("FRAMES_DIR") or tempfile.mkdtemp(prefix="desmos_")

if __name__ == "__main__":
    video, T0, dur, out = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
    SRC = os.path.join(TMP, "src")
    os.makedirs(SRC, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(T0), "-t", str(dur), "-i", video,
                    "-vf", f"fps={FPS},scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=0xfcc6dd",
                    os.path.join(SRC, "s%05d.png")], check=True)
    n = len(os.listdir(SRC))
    with Pool(os.cpu_count(), initializer=globals().__setitem__, initargs=("SRC", SRC)) as pool:
        list(pool.imap_unordered(work, range(n)))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", os.path.join(TMP, "d%05d.png"),
                    "-ss", str(T0), "-t", str(dur), "-i", video, "-map", "0:v", "-map", "1:a",
                    "-af", f"afade=t=in:d=0.2,afade=t=out:st={dur - 0.6}:d=0.6",
                    "-c:v", "libx264", "-crf", "20", "-preset", "slow", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
                    "-shortest", "-movflags", "+faststart", out], check=True)
    print("done", out)
