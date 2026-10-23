"""Риг по слоям: тело / левая рука / правая рука / голова (с подменой лица).
Руки вращаются в плечах (жесты в бит), голова — отдельный слой с наклоном вокруг шеи,
поверх — мягкая деформация волос, ахоге, юбки и пружина всего тела.

python3 rig_layers.py <img_open> <img_smile> <img_closed> <клип Daidaidai.mp4> out.mp4
"""
import math
import os
import subprocess
import sys
import tempfile
from multiprocessing import Pool

import numpy as np
from PIL import Image

import rig as RG
import layers as LY

TL = RG.TL
S2, S3, S4 = TL.B[10] + 0.93, 9.28, 12.97
LAY = {}


def prepare(p_open, p_smile, p_closed):
    RG.prepare(p_open, p_smile, p_closed)
    base = RG.G["A"][1]
    L = LY.build(base)
    body = LY.body_layer(base, L)
    LAY.update(L=L, body=Image.fromarray((np.clip(body, 0, 1) * 255).astype(np.uint8), "RGBA"))
    for s in ("L", "R"):
        m = L["arms"][s].astype(np.float32)
        LAY["arm" + s] = Image.fromarray((np.dstack([base, m]) * 255).astype(np.uint8), "RGBA")
        LAY["cap" + s] = LY.arm_with_cap(base, L["arms"][s], s)
    LAY["head_mask"] = L["head"].astype(np.float32)


def arm_target(t):
    """Целевые углы подъёма рук (градусы, 0 — вдоль тела, 180 — вверх)."""
    B = TL.B
    i = int(np.searchsorted(B, t, side="right")) - 1
    beat = (t - B[max(i, 0)]) / (B[1] - B[0])
    swing = math.sin(math.pi * (i + beat))
    pu = TL.pulse(t)
    if t < B[3]:
        return 4, 4
    if t < TL.DAI[0]:
        return 10 + 8 * swing, 10 - 8 * swing
    if t < TL.KIRAI:
        k = sum(1 for d in TL.DAI if t >= d) - 1
        up = 125 * math.exp(-(t - TL.DAI[k]) * 3)
        return (12 + up, 12) if k % 2 == 0 else (12, 12 + up)
    if t < S2:
        return 150, 150
    if t < S3:                                   # машет правой рукой
        return 12 + 6 * swing, 138 + 18 * swing
    if t < S4:                                   # руки в стороны, покачиваются
        return 55 + 15 * swing, 55 - 15 * swing
    if t < TL.BREAK0:                            # машет левой
        return 138 + 18 * swing, 14 + 6 * swing
    if t < TL.DROP:
        return 6, 6
    return 155 + 8 * pu, 155 + 8 * pu


def arm_angles(t):
    """Сглаживание целевых углов во времени — суставы не «телепортируются»."""
    ks = [arm_target(t - k * 0.025) for k in range(8)]
    w = [math.exp(-k * 0.45) for k in range(8)]
    sw = sum(w)
    return sum(a[0] * wk for a, wk in zip(ks, w)) / sw, sum(a[1] * wk for a, wk in zip(ks, w)) / sw


def compose(t):
    p, mouth, closed = RG.params(t)
    W, H = RG.G["W"], RG.G["H"]
    canvas = LAY["body"].copy()
    aL, aR = arm_angles(t)
    for s, ang in (("L", -aL), ("R", aR)):
        src = LAY[("cap" if abs(ang) > 105 else "arm") + s]
        arm = src.rotate(ang, resample=Image.BICUBIC, center=LY.SHOULDER[s])
        canvas.alpha_composite(arm)
    face = RG.face(mouth, closed)
    head = Image.fromarray((np.dstack([np.clip(face, 0, 1), LAY["head_mask"]]) * 255).astype(np.uint8), "RGBA")
    head = head.rotate(p["head"], resample=Image.BICUBIC, center=LY.NECK_PIVOT,
                       translate=(p["hx"], p["hy"]))
    canvas.alpha_composite(head)
    arr = np.asarray(canvas, np.float32) / 255
    # общая деформация (пружина, покачивание, волосы, ахоге, юбка) — голова уже повёрнута слоем
    q = dict(p, head=0.0, hx=0.0, hy=0.0)
    img, al = RG.warp(arr[..., :3], arr[..., 3], q)
    return img, al


def render(fi):
    t = TL.T0 + fi / RG.FPS
    img, al = compose(t)
    return RG.render_from(t, img, al)


def init(paths):
    prepare(*paths)


def work(fi):
    p = os.path.join(TMP, f"l{fi:05d}.jpg")
    if not os.path.exists(p):
        render(fi).save(p, quality=93)
    return fi


TMP = os.environ.get("FRAMES_DIR") or tempfile.mkdtemp(prefix="rigl_")

if __name__ == "__main__":
    paths, song, out = sys.argv[1:4], sys.argv[4], sys.argv[5]
    os.makedirs(TMP, exist_ok=True)
    n = int((TL.T1 - TL.T0) * RG.FPS)
    with Pool(os.cpu_count(), initializer=init, initargs=(paths,)) as pool:
        list(pool.imap_unordered(work, range(n)))
    dur = n / RG.FPS
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(RG.FPS), "-i", os.path.join(TMP, "l%05d.jpg"),
                    "-ss", str(43.0 + TL.T0), "-t", str(dur), "-i", song, "-map", "0:v", "-map", "1:a",
                    "-af", f"afade=t=in:d=0.2,afade=t=out:st={dur - 0.8}:d=0.8",
                    "-c:v", "libx264", "-crf", "21", "-preset", "slow", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out], check=True)
    print("done", out)
