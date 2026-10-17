"""Тестовый эдит: кадры «кот-рыцарь против дракона» под припев Daidaidai (雨良).

Монтаж в бит, наезды-удары, тряска, вспышки, искры, лучи, цветокор, свечение, надписи.
python3 cat_edit.py <папка_кадров_исходника> <клип Daidaidai.mp4> out.mp4
"""
import json
import math
import os
import subprocess
import sys
import tempfile
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "daikirai_demo"))
from flat import Fonts  # noqa: E402

W, H, FPS = 1280, 720, 30
SRC_FPS = 30
AN = json.load(open(os.path.join(HERE, "..", "daikirai_demo", "analysis.json")))
B = np.array(AN["beats"])                 # доли от 0:43.0 трека
T0, T1 = 0.6, 19.0                        # окно эдита (сек от 0:43.0)
E8 = (B[1] - B[0]) / 2
DAI = [B[6] + E8 * k for k in range(1, 6)]
KIRAI = B[9]
BREAK0, DROP = 15.30, 16.90
SRC = None

# план: (начало, конец, источник_с, источник_по, наезд_от, наезд_до, сдвиг_x, сдвиг_y, вращение°, якорь_y)
SHOTS = [
    (T0, B[3], 4.8, 6.0, 1.00, 1.18, 0, 0, 0, 0.5),            # тьма, глаз дракона
    (B[3], DAI[0], 2.5, 3.9, 1.02, 1.10, 20, 0, 0, 0.42),      # кот во дворце
    (DAI[0], DAI[1], 6.4, 6.6, 1.05, 1.12, 0, 0, -3, 0.4),     # «ダイ» × 5 — нарезка
    (DAI[1], DAI[2], 14.8, 15.0, 1.05, 1.12, 0, 0, 3, 0.5),
    (DAI[2], DAI[3], 4.3, 4.5, 1.05, 1.15, 0, 0, -2, 0.5),
    (DAI[3], DAI[4], 20.0, 20.2, 1.02, 1.10, 0, 0, 2, 0.55),
    (DAI[4], KIRAI, 9.8, 10.0, 1.08, 1.15, 0, 0, -4, 0.45),
    (KIRAI, B[10] + 0.93, 11.4, 12.5, 1.08, 1.00, -10, 0, 0, 0.45),  # «キライ» — удар
    (B[10] + 0.93, B[15], 6.3, 7.5, 1.00, 1.12, 30, -10, 0, 0.42),   # замок с драконом
    (B[15], B[17], 9.7, 10.9, 1.06, 1.00, 0, 0, 0, 0.42),       # кот с мечом
    (B[17], B[19], 8.4, 9.1, 1.00, 1.08, -20, 0, 0, 0.5),       # поле
    (B[19], B[21], 15.9, 17.2, 1.00, 1.10, 0, 10, 0, 0.4),      # полёт
    (B[21], B[23], 17.3, 18.3, 1.02, 1.10, 0, 0, 2, 0.5),
    (B[23], B[27], 18.5, 19.7, 1.02, 1.15, 0, 0, 0, 0.55),      # светящийся меч
    (B[27], B[31], 19.8, 21.2, 1.00, 1.10, 20, 0, -2, 0.55),    # дракон ревёт
    (B[31], BREAK0, 12.5, 13.2, 1.08, 1.15, 0, 0, 0, 0.45),
    (BREAK0, DROP, 22.3, 24.0, 1.00, 1.25, 80, -20, 0, 0.62),    # брейк: медленный наезд на героя
    (DROP, T1 + 0.01, 25.4, 28.0, 1.10, 1.00, 60, 0, 0, 0.62),  # «あぁ» — финальный удар
]


def beat_pulse(t, decay=7.0):
    i = int(np.searchsorted(B, t, side="right")) - 1
    if i < 0:
        return 0.0
    if BREAK0 <= t < DROP:
        return 0.0
    return math.exp(-(t - B[i]) * decay)


def src_frame(s):
    idx = int(np.clip(round(s * SRC_FPS), 0, 855)) + 1
    return Image.open(os.path.join(SRC, f"f{idx:04d}.jpg")).convert("RGB")


def place(img, zoom, dx, dy, rot, anchor_y):
    """Вписать кадр 960×720 в 1280×720 с наездом, сдвигом и вращением (одна аффинная операция)."""
    sw, sh = img.size
    base = W / sw                                   # ширина по кадру
    s = base * zoom
    cx_s, cy_s = sw / 2 - dx / s, sh * anchor_y + (sh / 2 - sh * anchor_y) * 0 - dy / s
    cy_s = sh * anchor_y + (sh / 2 - sh * anchor_y) * 0.0 - dy / s
    a = math.radians(rot)
    c, si = math.cos(a) / s, math.sin(a) / s
    # out(x, y) → src: [c -si; si c] · (x − W/2, y − H/2) + (cx, cy)
    m = (c, -si, cx_s - c * W / 2 + si * H / 2, si, c, cy_s - si * W / 2 - c * H / 2)
    return img.transform((W, H), Image.AFFINE, m, resample=Image.BICUBIC)


def grade(a):
    """Цветокор: контраст, тёплые света / холодные тени, насыщенность."""
    lum = a.mean(axis=2, keepdims=True)
    a = a * 0.88
    a = (a - 0.45) * 1.10 + 0.45
    hi = np.clip((lum - 0.5) * 2, 0, 1)
    a = a * (1 + hi * np.array([0.06, 0.0, -0.07])) * (1 + (1 - hi) * np.array([-0.04, 0.0, 0.06]))
    g = a.mean(axis=2, keepdims=True)
    a = g + (a - g) * 1.18
    return np.clip(a, 0, 1)


def bloom(a, amount=0.45):
    small = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).resize((W // 4, H // 4), Image.BILINEAR)
    s = np.asarray(small, np.float32) / 255
    s = np.clip(s - 0.86, 0, 1) / 0.14
    b = Image.fromarray((s * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(10)).resize((W, H), Image.BILINEAR)
    return a + np.asarray(b, np.float32) / 255 * amount


def embers(t, density=1.0, seed=1):
    """Искры-угольки: детерминированные частицы, летят вверх с дрожанием."""
    layer = Image.new("RGB", (W, H), (0, 0, 0))
    d = ImageDraw.Draw(layer)
    r = np.random.default_rng(seed)
    n = int(90 * density)
    for k in range(n):
        x0, life, sp, ph, size = r.uniform(0, W), r.uniform(1.2, 2.6), r.uniform(120, 320), r.uniform(0, 10), r.uniform(1.5, 4)
        age = (t + ph) % life
        y = H + 20 - age * sp
        x = x0 + 40 * math.sin(age * 3 + k) + age * 30
        fade = math.sin(math.pi * age / life)
        col = (int(255 * fade), int(150 * fade), int(40 * fade))
        d.line([(x, y), (x - 4, y + size * 3)], fill=col, width=max(1, int(size)))
    layer = layer.filter(ImageFilter.GaussianBlur(1.2))
    return np.asarray(layer, np.float32) / 255


def rays(t, strength):
    """Лучи света из верхнего угла (брейк)."""
    layer = Image.new("L", (W // 2, H // 2), 0)
    d = ImageDraw.Draw(layer)
    ox, oy = W * 0.78 / 2, -20
    for k in range(9):
        a = math.radians(95 + k * 9 + 3 * math.sin(t * 0.7 + k))
        w = math.radians(2.5 + (k % 3))
        d.polygon([(ox, oy), (ox + 900 * math.cos(a - w), oy + 900 * math.sin(a - w)),
                   (ox + 900 * math.cos(a + w), oy + 900 * math.sin(a + w))], fill=int(110 + 60 * (k % 2)))
    layer = layer.filter(ImageFilter.GaussianBlur(14)).resize((W, H), Image.BILINEAR)
    return (np.asarray(layer, np.float32) / 255 * strength)[..., None] * np.array([1.0, 0.92, 0.75])


def stamp(img, s, x, y, px, color, outline, t_since, rot=0.0):
    if t_since < 0:
        return
    sc = 1.0 + 0.7 * math.exp(-t_since * 16)
    size = int(px * sc)
    glyphs = [(ch, Fonts.font_for("dela-gothic-one", ch, size)) for ch in s]
    wsum = sum(f.getlength(ch) for ch, f in glyphs)
    pad = 20
    lay = Image.new("RGBA", (int(wsum) + pad * 2, int(size * 1.4) + pad * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    cx = pad
    for ch, f in glyphs:
        d.text((cx, pad), ch, font=f, fill=color, stroke_width=max(3, size // 14), stroke_fill=outline)
        cx += f.getlength(ch)
    lay = lay.rotate(rot, expand=True, resample=Image.BICUBIC)
    img.alpha_composite(lay, (int(x - lay.width / 2), int(y - lay.height / 2)))


def shot_at(t):
    for sh in SHOTS:
        if sh[0] <= t < sh[1]:
            return sh
    return SHOTS[-1]


def base_frame(t, sh, extra_zoom=1.0, jitter=(0, 0, 0)):
    t0, t1, s0, s1, z0, z1, dx, dy, rot, ay = sh
    k = (t - t0) / max(1e-6, t1 - t0)
    s = s0 + (s1 - s0) * k
    ease = 1 - (1 - k) ** 2
    z = (z0 + (z1 - z0) * ease) * extra_zoom
    img = src_frame(s)
    return place(img, z, dx * ease + jitter[0], dy * ease + jitter[1], rot + jitter[2], ay)


def render(fi):
    t = T0 + fi / FPS
    sh = shot_at(t)
    pulse = beat_pulse(t)
    since_cut = t - sh[0]
    r = np.random.default_rng(fi)
    shake_amt = 0.0
    if t >= KIRAI and t < KIRAI + 0.6:
        shake_amt = 18 * math.exp(-(t - KIRAI) * 5)
    if t >= DROP:
        shake_amt = max(shake_amt, 22 * math.exp(-(t - DROP) * 3))
    if B[3] <= t < BREAK0:
        shake_amt = max(shake_amt, 6 * pulse)
    jit = (r.normal(0, 1) * shake_amt, r.normal(0, 1) * shake_amt, r.normal(0, 0.15) * shake_amt * 0.2)
    zoom_punch = 1 + 0.05 * pulse if B[3] <= t < BREAK0 else 1.0
    # размытие наезда в первые кадры склейки и на ударах: среднее нескольких масштабов
    blur_n = 4 if since_cut < 0.07 or pulse > 0.8 else 1
    acc = None
    for j in range(blur_n):
        ez = zoom_punch * (1 + 0.035 * j)
        f = np.asarray(base_frame(t, sh, ez, jit), np.float32) / 255
        acc = f if acc is None else acc + f
    a = acc / blur_n
    a = grade(a)
    # брейк: приглушённый цвет, лучи; дроп — тёплый свет
    if BREAK0 <= t < DROP:
        k = (t - BREAK0) / (DROP - BREAK0)
        g = a.mean(axis=2, keepdims=True)
        a = g + (a - g) * (1 - 0.45 * (1 - k))
        a = a + rays(t, 0.25 + 0.5 * k)
    if t >= DROP or (11.4 <= sh[2] <= 15.0) or sh[2] >= 19.8:
        a = a + embers(t, 1.3 if t >= DROP else 0.7) * 0.9
    a = bloom(a, 0.35 if t >= DROP else 0.22)
    # хроматическая аберрация на ударах
    ca = int(round(2 + 10 * max(pulse if B[3] <= t < BREAK0 else 0, math.exp(-max(0, t - KIRAI) * 8) if t >= KIRAI else 0,
                               math.exp(-max(0, t - DROP) * 6) if t >= DROP else 0)))
    if ca > 1:
        a[..., 0] = np.roll(a[..., 0], ca, axis=1)
        a[..., 2] = np.roll(a[..., 2], -ca, axis=1)
    # вспышки: вход бочки, «キライ», дроп, каждая «ダイ» слегка
    fl = 0.0
    for tt, amp, dec in [(B[3], 0.8, 7), (KIRAI, 0.7, 8), (DROP, 0.9, 5)] + [(d_, 0.12, 16) for d_ in DAI]:
        if t >= tt:
            fl = max(fl, amp * math.exp(-(t - tt) * dec))
    if t < B[3]:                               # нарастание к удару
        fl = max(fl, 0.5 * max(0, (t - (B[3] - 0.35)) / 0.35) ** 2)
        a = a * (0.55 + 0.45 * abs(math.sin(t * 23)) ** 0.2)          # мерцание во тьме
    a = a * (1 - fl) + fl
    # виньетка, зерно, «кино»-полосы
    yy, xx = np.mgrid[0:H, 0:W]
    v = 1 - 0.38 * np.clip(np.hypot((xx - W / 2) / W, (yy - H / 2) / H) - 0.25, 0, 1) ** 1.5
    a = a * v[..., None] + r.normal(0, 0.018, (H, W, 1))
    out = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).convert("RGBA")
    d = ImageDraw.Draw(out)
    d.rectangle([0, 0, W, 42], fill=(0, 0, 0, 255))
    d.rectangle([0, H - 42, W, H], fill=(0, 0, 0, 255))
    # надписи в бит
    pos = [(250, 200, -12), (1030, 190, 10), (230, 470, 8), (1050, 480, -10), (640, 140, 4)]
    for k, dt in enumerate(DAI):
        if dt <= t < KIRAI + 0.9:
            x, y, rr = pos[k]
            stamp(out, "ダイ", x, y, 120, (255, 255, 255, 255) if k % 2 == 0 else (255, 214, 70, 255), (20, 10, 30, 255), t - dt, rr)
    if KIRAI <= t < KIRAI + 0.9:
        stamp(out, "キライ", 640, 560, 150, (255, 50, 80, 255), (255, 255, 255, 255), t - KIRAI, -3)
    if t >= DROP + 0.15:
        al = min(1.0, (t - DROP - 0.15) / 0.3)
        stamp(out, "あぁ", 1040, 160, int(110), (255, 255, 255, int(255 * al)), (20, 10, 30, int(255 * al)), t - DROP - 0.15, 6)
    # затемнение в конце
    if t > T1 - 0.6:
        k = (t - (T1 - 0.6)) / 0.6
        out = Image.blend(out, Image.new("RGBA", (W, H), (0, 0, 0, 255)), min(1, k))
    return out.convert("RGB")


def work(fi):
    p = os.path.join(TMP, f"e{fi:05d}.jpg")
    if not os.path.exists(p):
        render(fi).save(p, quality=93)
    return fi


TMP = os.environ.get("FRAMES_DIR") or tempfile.mkdtemp(prefix="catedit_")

if __name__ == "__main__":
    SRC = sys.argv[1]
    song, out = sys.argv[2], sys.argv[3]
    os.makedirs(TMP, exist_ok=True)
    n = int((T1 - T0) * FPS)
    with Pool(os.cpu_count(), initializer=globals().__setitem__, initargs=("SRC", SRC)) as pool:
        list(pool.imap_unordered(work, range(n)))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", os.path.join(TMP, "e%05d.jpg"),
                    "-ss", str(43.0 + T0), "-t", str(T1 - T0), "-i", song, "-map", "0:v", "-map", "1:a",
                    "-af", f"afade=t=in:d=0.2,afade=t=out:st={T1 - T0 - 0.8}:d=0.8",
                    "-c:v", "libx264", "-crf", "20", "-preset", "slow", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out], check=True)
    print("done", out)
