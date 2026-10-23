"""2D-риг из одной картинки (+ 2 варианта лица): мимика подменой частей,
движение — поле смещений (как деформеры Live2D): голова вокруг шеи, волосы и ахоге
с запаздыванием, дыхание, пружина в бит, юбка и рукава. Вертикальное видео 9:16.

python3 rig.py <img_open_mouth> <img_smile> <img_eyes_closed> <клип Daidaidai.mp4> out.mp4
"""
import math
import os
import subprocess
import sys
import tempfile
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy import ndimage

import importlib.util
HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("tl", os.path.join(HERE, "..", "y2k", "timeline.py"))
TL = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(TL)

OW, OH, FPS = 1080, 1920, TL.FPS
FJP = os.path.join(HERE, "..", "daikirai_demo", "fonts", "dela-gothic-one-japanese-400-normal.ttf")
NECK = (565.0, 628.0)
FEET = (565.0, 1900.0)
G = {}


def smooth(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def prepare(p_open, p_smile, p_closed):
    A = [np.asarray(Image.open(p).convert("RGB"), np.float32) / 255 for p in (p_open, p_smile, p_closed)]
    H, W = A[0].shape[:2]
    # маски частей лица по разнице картинок
    mouth = (np.abs(A[0] - A[1]).sum(2) > 0.12)
    eyes = (np.abs(A[1] - A[2]).sum(2) > 0.12)
    eyes[640:] = False
    mouth = ndimage.binary_dilation(mouth, iterations=4)
    eyes = ndimage.binary_dilation(eyes, iterations=4)
    soft = lambda m: ndimage.gaussian_filter(m.astype(np.float32), 1.5)[..., None]
    # фон → прозрачность: белые области, связанные с краем кадра
    white = A[1].min(2) > 0.93
    lab, _ = ndimage.label(white)
    border = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])) - {0}
    bg = np.isin(lab, list(border))
    alpha = 1 - ndimage.gaussian_filter(bg.astype(np.float32), 0.8)
    # «полуоткрытый» рот: открытый рот, сжатый по вертикали к верхней губе
    ys, xs = np.nonzero(np.abs(A[0] - A[1]).sum(2) > 0.12)
    y0, y1, x0, x1 = ys.min() - 3, ys.max() + 4, xs.min() - 3, xs.max() + 4
    patch = A[0][y0:y1, x0:x1]
    half = A[1].copy()
    ph = Image.fromarray((patch * 255).astype(np.uint8)).resize((x1 - x0, max(4, int((y1 - y0) * 0.55))), Image.BICUBIC)
    pa = np.asarray(ph, np.float32) / 255
    mm = (np.abs(A[0] - A[1]).sum(2) > 0.12)[y0:y1, x0:x1].astype(np.uint8) * 255
    mm = np.asarray(Image.fromarray(mm).resize(ph.size, Image.BILINEAR).filter(ImageFilter.GaussianBlur(1)), np.float32)[..., None] / 255
    yy0 = y0 + 2
    half[yy0:yy0 + pa.shape[0], x0:x1] = half[yy0:yy0 + pa.shape[0], x0:x1] * (1 - mm) + pa * mm
    G.update(A=A, H=H, W=W, mouth=soft(mouth), eyes=soft(eyes), alpha=alpha, half=half)


def face(mouth_state, eyes_closed):
    A = G["A"]
    base = {"open": A[0], "half": G["half"], "closed": A[1]}[mouth_state]
    if eyes_closed:
        base = base * (1 - G["eyes"]) + A[2] * G["eyes"]
    return base


def warp(img, alpha, p):
    """Обратное отображение: для каждого пикселя результата — откуда брать в исходнике."""
    H, W = G["H"], G["W"]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    sx, sy = xx.copy(), yy.copy()
    # тело: пружина (сжатие к ногам), покачивание вокруг стоп
    s = 1 - 0.018 * p["bounce"]
    sy = FEET[1] + (sy - FEET[1]) / s
    a = -math.radians(p["body"])
    c, si = math.cos(a), math.sin(a)
    dx, dy = sx - FEET[0], sy - FEET[1]
    sx, sy = FEET[0] + c * dx - si * dy, FEET[1] + si * dx + c * dy
    # дыхание грудной клетки
    br = smooth(1150, 900, sy) * smooth(600, 700, sy)
    sy = sy + br * p["breath"] * 4
    # юбка и рукава
    skirt = smooth(1110, 1160, sy) * smooth(1290, 1240, sy)
    sx = sx - skirt * (sx - 565) * 0.03 * math.sin(p["ph"] * 2) - skirt * 6 * math.sin(p["ph"] * 2 - 0.6)
    sleeve = smooth(170, 210, np.abs(sx - 565)) * smooth(690, 760, sy) * smooth(1260, 1200, sy)
    sx = sx - sleeve * np.sign(sx - 565) * 5 * math.sin(p["ph"] * 2 + 0.4) * smooth(700, 1240, sy)
    # голова: поворот вокруг шеи + смещение, переход на шее
    wh = smooth(668, 600, sy)
    a = -math.radians(p["head"])
    c, si = math.cos(a), math.sin(a)
    dx, dy = sx - NECK[0] - p["hx"], sy - NECK[1] - p["hy"]
    hx_, hy_ = NECK[0] + c * dx - si * dy, NECK[1] + si * dx + c * dy
    sx, sy = sx + wh * (hx_ - sx), sy + wh * (hy_ - sy)
    # волосы по бокам: качаются с запаздыванием, сильнее к кончикам
    side = smooth(110, 230, np.abs(sx - 565)) * smooth(380, 560, sy) * wh
    sx = sx - side * p["hair"] * smooth(380, 650, sy)
    # ахоге
    wa = smooth(285, 230, sy) * wh
    a = -math.radians(p["aho"])
    c, si = math.cos(a), math.sin(a)
    dx, dy = sx - 560, sy - 285
    sx, sy = sx + wa * (560 + c * dx - si * dy - sx), sy + wa * (285 + si * dx + c * dy - sy)
    out = np.stack([ndimage.map_coordinates(img[..., k], [sy, sx], order=1, mode="nearest") for k in range(3)], -1)
    al = ndimage.map_coordinates(alpha, [sy, sx], order=1, mode="constant", cval=0)
    return out, al


# ---------------- анимация ----------------

BLINKS = [2.4, 5.6, 8.1, 11.0, 13.6, 16.2]


def params(t):
    i = int(np.searchsorted(TL.B, t, side="right")) - 1
    ph = math.pi * (t - TL.B[max(i, 0)]) / (TL.B[1] - TL.B[0]) + math.pi * max(i, 0)
    pu = TL.pulse(t)
    energy = 0.35 if t < TL.B[3] or TL.BREAK0 <= t < TL.DROP else 1.0
    side = 1 if i % 2 else -1
    p = dict(
        ph=ph, bounce=pu * energy,
        body=1.4 * energy * math.sin(ph / 2),
        head=5.0 * energy * math.sin(ph / 2 - 0.5) + (3 * math.sin(t * 1.3) if energy < 1 else 0),
        hx=6 * energy * math.sin(ph / 2 - 0.5), hy=-10 * pu * energy,
        breath=math.sin(t * 2 * math.pi / 3.0),
        hair=10 * energy * math.sin(ph / 2 - 1.3) + 3 * math.sin(t * 2.1),
        aho=14 * math.sin(ph / 2 - 1.8) * energy + 4 * math.sin(t * 3),
    )
    e = TL.env_at(t)
    mouth = "open" if e > 0.5 else ("half" if e > 0.24 else "closed")
    closed = any(0 <= t - b < 0.13 for b in BLINKS)
    if TL.KIRAI <= t < TL.KIRAI + 0.45 or (TL.BREAK0 + 0.3 <= t < TL.DROP) or t >= TL.DROP + 0.6:
        closed = True
    return p, mouth, closed


def camera(t):
    """(центр_x, центр_y, масштаб) в координатах картинки; масштаб = пикселей кадра на пиксель картинки."""
    full = (565, 1020, 0.92)
    bust = (565, 640, 1.9)
    face_ = (565, 520, 3.0)
    if t < TL.B[3]:
        k = (t - TL.T0) / (TL.B[3] - TL.T0)
        return (565, 600 + 300 * k, 2.6 - 1.6 * k)
    if t < TL.DAI[0]:
        return full
    if t < TL.KIRAI:
        k = sum(1 for d in TL.DAI if t >= d) - 1
        return [bust, face_, bust, face_, bust][k]
    if t < TL.KIRAI + 0.9:
        return (565, 600, 2.2 + 0.4 * math.exp(-(t - TL.KIRAI) * 6))
    if t < 9.28:
        return full
    if t < 12.97:
        k = (t - 9.28) / (12.97 - 9.28)
        return (565, 700 - 80 * k, 1.45 + 0.2 * k)
    if t < TL.BREAK0:
        return full
    if t < TL.DROP:
        k = (t - TL.BREAK0) / (TL.DROP - TL.BREAK0)
        return (565, 560, 1.6 + 1.6 * k * k)
    k = min(1.0, (t - TL.DROP) / 0.8)
    return (565, 900 + 120 * k, 1.5 - 0.55 * k)


def background(t):
    yy, xx = np.mgrid[0:OH, 0:OW].astype(np.float32)
    u = yy / OH
    top, bot = np.array([0.80, 0.86, 0.97]), np.array([0.96, 0.95, 1.0])
    a = top * (1 - u[..., None]) + bot * u[..., None]
    ph = (xx / OW * 1.4 - yy / OH * 0.9 + t * 0.15) * math.pi
    holo = np.stack([0.5 + 0.5 * np.sin(ph), 0.5 + 0.5 * np.sin(ph + 2.1), 0.5 + 0.5 * np.sin(ph + 4.2)], -1)
    a = a * 0.9 + holo * 0.1
    im = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).convert("RGBA")
    d = ImageDraw.Draw(im, "RGBA")
    r = np.random.default_rng(3)
    for k in range(14):                                   # пузыри поднимаются
        x = r.uniform(60, OW - 60) + 30 * math.sin(t * 0.8 + k)
        sp = r.uniform(40, 90)
        y = OH + 100 - ((t * sp + r.uniform(0, OH + 200)) % (OH + 200))
        rr = r.uniform(20, 70)
        d.ellipse((x - rr, y - rr, x + rr, y + rr), outline=(255, 255, 255, 170), width=3)
        d.arc((x - rr * 0.7, y - rr * 0.7, x + rr * 0.4, y + rr * 0.4), 200, 260, fill=(255, 255, 255, 220), width=4)
    rays = Image.new("RGBA", (OW, OH), (0, 0, 0, 0))
    dr = ImageDraw.Draw(rays)
    for k in range(9):                                    # лучи — отдельным полупрозрачным слоем
        a0 = math.radians(-60 + k * 15 + 4 * math.sin(t * 0.5 + k))
        dr.polygon([(OW / 2, -80), (OW / 2 + 2200 * math.sin(a0 - 0.03), 2200 * math.cos(a0 - 0.03)),
                    (OW / 2 + 2200 * math.sin(a0 + 0.03), 2200 * math.cos(a0 + 0.03))], fill=(255, 255, 255, 40))
    im.alpha_composite(rays)
    return im


def sparkles(im, t, n=16):
    d = ImageDraw.Draw(im, "RGBA")
    r = np.random.default_rng(9)
    for k in range(n):
        x, y = r.uniform(40, OW - 40), r.uniform(40, OH - 40)
        s = (6 + 16 * max(0.0, math.sin(t * 2.4 + r.uniform(0, 6.28)))) * (1 + 0.7 * TL.pulse(t))
        if s < 8:
            continue
        al = int(235 * min(1, (s - 7) / 9))
        for ang in (0, math.pi / 2):
            dx, dy = math.cos(ang) * s, math.sin(ang) * s
            d.line([(x - dx, y - dy), (x + dx, y + dy)], fill=(255, 255, 255, al), width=3)


def stamp(im, s, x, y, px, t_since, rot, col=(255, 255, 255), out=(70, 120, 200)):
    if t_since < 0:
        return
    sc = 1 + 0.6 * math.exp(-t_since * 14)
    f = ImageFont.truetype(FJP, int(px * sc))
    lay = Image.new("RGBA", (int(px * sc * len(s) * 1.2 + 60), int(px * sc * 1.6)), (0, 0, 0, 0))
    ImageDraw.Draw(lay).text((30, 10), s, font=f, fill=col + (255,), stroke_width=max(4, px // 12), stroke_fill=out + (255,))
    lay = lay.rotate(rot, expand=True, resample=Image.BICUBIC)
    im.alpha_composite(lay, (int(x - lay.width / 2), int(y - lay.height / 2)))


def render(fi):
    t = TL.T0 + fi / FPS
    p, mouth, closed = params(t)
    img, al = warp(face(mouth, closed), G["alpha"], p)
    return render_from(t, img, al)


def render_from(t, img, al):
    """Кадр из готового (уже анимированного) персонажа: фон, камера, надписи, вспышки."""
    ch = Image.fromarray((np.dstack([np.clip(img, 0, 1), al]) * 255).astype(np.uint8), "RGBA")
    cx, cy, sc = camera(t)
    sc *= 1 + 0.03 * TL.pulse(t)
    out = background(t)
    # тень на полу
    sh = Image.new("RGBA", (OW, OH), (0, 0, 0, 0))
    fx, fy = OW / 2 + (FEET[0] - cx) * sc, OH / 2 + (FEET[1] - 10 - cy) * sc
    ImageDraw.Draw(sh).ellipse((fx - 180 * sc, fy - 22 * sc, fx + 180 * sc, fy + 22 * sc), fill=(90, 110, 160, 70))
    out.alpha_composite(sh.filter(ImageFilter.GaussianBlur(12)))
    # персонаж в кадр
    W, H = G["W"], G["H"]
    chs = ch.resize((max(1, int(W * sc)), max(1, int(H * sc))), Image.BICUBIC if sc < 1.2 else Image.LANCZOS)
    ox, oy = int(OW / 2 - cx * sc), int(OH / 2 - cy * sc)
    # белая обводка-«наклейка» вокруг силуэта
    a = chs.split()[3].filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.GaussianBlur(1))
    out.paste(chs, (ox, oy), chs)
    sparkles(out, t)
    # надписи в бит
    pos = [(250, 330, -12), (830, 360, 10), (230, 1500, 8), (850, 1530, -8), (540, 250, 4)]
    for k, dt in enumerate(TL.DAI):
        if dt <= t < TL.KIRAI + 0.05:
            x, y, r = pos[k]
            stamp(out, "ダイ", x, y, 150, t - dt, r)
    if TL.KIRAI <= t < TL.KIRAI + 0.9:
        stamp(out, "ダイキライ", OW / 2, 1560, 170, t - TL.KIRAI, -4, col=(255, 255, 255), out=(230, 70, 110))
    if t >= TL.DROP + 0.2:
        stamp(out, "あぁ", 800, 330, 160, t - TL.DROP - 0.2, 8)
    # вспышки
    fl = 0.0
    for tt, amp, dec in TL.FLASHES:
        if t >= tt:
            fl = max(fl, amp * math.exp(-(t - tt) * dec))
    if fl > 0.01:
        out = Image.blend(out, Image.new("RGBA", (OW, OH), (255, 255, 255, 255)), min(0.9, fl * 0.8))
    if t > TL.T1 - 0.7:
        out = Image.blend(out, Image.new("RGBA", (OW, OH), (245, 247, 252, 255)), min(1.0, (t - (TL.T1 - 0.7)) / 0.7))
    return out.convert("RGB")


def init(paths):
    prepare(*paths)


def work(fi):
    p = os.path.join(TMP, f"g{fi:05d}.jpg")
    if not os.path.exists(p):
        render(fi).save(p, quality=93)
    return fi


TMP = os.environ.get("FRAMES_DIR") or tempfile.mkdtemp(prefix="rig_")

if __name__ == "__main__":
    paths, song, out = sys.argv[1:4], sys.argv[4], sys.argv[5]
    os.makedirs(TMP, exist_ok=True)
    n = int((TL.T1 - TL.T0) * FPS)
    with Pool(os.cpu_count(), initializer=init, initargs=(paths,)) as pool:
        list(pool.imap_unordered(work, range(n)))
    dur = n / FPS
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", os.path.join(TMP, "g%05d.jpg"),
                    "-ss", str(43.0 + TL.T0), "-t", str(dur), "-i", song, "-map", "0:v", "-map", "1:a",
                    "-af", f"afade=t=in:d=0.2,afade=t=out:st={dur - 0.8}:d=0.8",
                    "-c:v", "libx264", "-crf", "21", "-preset", "slow", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out], check=True)
    print("done", out)
