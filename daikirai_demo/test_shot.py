"""Тестовый план: «大体 アナタのコト アタシ / ダイダイダイダイダイキライ».

Ключевые рисунки поз + по одной промежуточной фазе, анимация «по двойкам».
python3 test_shot.py <клип_с_треком.mp4> out.mp4
"""
import math
import os
import subprocess
import sys
import tempfile
from multiprocessing import Pool

import numpy as np
from flat import Canvas, Tex, Xf, hexc, LINE
import chara
import drawings as D
import lyrics as L
import scenes as S

T0, T1 = 1.25, 5.6          # секунды от начала отрезка (0:43)
FPS, TWOS = 24, 2
E8 = (L.BEATS[1] - L.BEATS[0]) / 2
DAI = [L.BEATS[6] + E8 * k for k in range(1, 6)]   # «ダイ» восьмыми
KIRAI = L.BEATS[9]

A = (D.arm_down(-1), D.arm_down(1))
B = (D.arm_down(-1), D.arm_point(1))
Cc = (D.arm_chest(-1), D.arm_down(1))
PRE = (D.arm_pre_cross(-1), D.arm_pre_cross(1))
E = (D.arm_cross(-1, True), D.arm_cross(1, False))


def mix(p, q, t):
    return (D.lerp_arm(p[0], q[0], t), D.lerp_arm(p[1], q[1], t))


# (с какого момента, руки, кто спереди, выражение)
TIMELINE = [
    (0.00, A, "R", "sing"),
    (2.10, mix(A, B, 0.5), "L", "sing"),
    (2.18, B, "L", "point"),
    (2.98, mix(B, Cc, 0.5), "R", "sing"),
    (3.06, Cc, "R", "self"),
    (3.40, mix(Cc, A, 0.5), "R", "sing"),
    (3.48, A, "R", "dai"),
    (KIRAI - 0.17, mix(A, PRE, 0.6), "R", "dai"),
    (KIRAI - 0.09, PRE, "R", "dai"),
    (KIRAI, E, "R", "kirai"),
]


def key_at(t):
    cur = TIMELINE[0]
    for row in TIMELINE:
        if t >= row[0]:
            cur = row
    return cur


def pose_for(t):
    start, arms, front, mood = key_at(t)
    p, pulse = S.base_pose(t, 1.0)
    v, op = L.mouth_at(t)
    p.update(body_fn=D.torso, arms_fn=D.arms_fn(arms[0], arms[1], front), armL=None, armR=None)
    if mood == "point":
        p.update(eyes="open", brow="smug", look=(3, 2), head_rot=-4)
    elif mood == "self":
        p.update(eyes="half", brow="smug", head_rot=6)
    elif mood == "dai":
        # кивок на каждое «ダイ»
        k = max([i for i, d in enumerate(DAI) if t >= d] or [-1])
        ph = (t - DAI[k]) / E8 if k >= 0 else 1
        p["hy"] = -10 * math.exp(-ph * 3) if k >= 0 else 0
        p["head_rot"] = (5 if k % 2 else -5) * math.exp(-ph * 1.5)
        p.update(eyes="angry", brow="angry")
        if op > 0.12:
            p["mouth"] = dict(kind="grin", open=op)
    elif mood == "kirai":
        p.update(eyes="angry", brow="angry", head_rot=-3, hy=-4)
        p["mouth"] = dict(kind="grin", open=max(op, 0.6)) if t < KIRAI + 0.45 else p["mouth"]
    return p, pulse


def background(cv, t):
    if t < DAI[0]:
        i, ph = L.beat_phase(t)
        S.bg_split(cv, S.RED, S.BLUE_D, slope=0.5)
        S.rings(cv, 640, 300, 7, S.BLACK, S.WHITE, r0=40, step=58, pulse=math.exp(-ph * 3))
        cv.halftone(S.RED_D, 11, 0.6, density=lambda x, y: np.clip(1 - x / 900, 0, 1) * 0.6)
    else:
        k = max([i for i, d in enumerate(DAI) if t >= d] or [0])
        flip = k % 2 == 0 if t < KIRAI else True
        cv.img[:] = np.asarray(S.RED_D if flip else S.BLUE_D, np.float32)
        cv.halftone(S.RED if flip else S.BLUE, 13, 0.4, density=lambda x, y: np.clip(np.hypot(x - 640, y - 330) / 700, 0, 1))
        if t >= KIRAI:
            # концентрические «удары» за спиной
            for r in range(6, 0, -1):
                rr = 90 * r + (t - KIRAI) * 300
                cv.fill(np.array([(640 + rr * math.cos(a), 300 + rr * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 80)]),
                        S.BLACK if r % 2 else S.RED, alpha=0.5, jitter=False)


def overlays(cv, t, fx):
    pos = [(210, 150, -0.15), (1070, 150, 0.12), (190, 450, 0.1), (1090, 450, -0.12), (230, 300, 0.05)]
    for k, d in enumerate(DAI):
        if t >= d:
            x, y, r = pos[k]
            S.stamp(cv, "ダイ", x, y, 120, S.WHITE if k % 2 == 0 else S.YELLOW, t - d, r,
                    out=S.RED_D if k % 2 else S.BLUE_D)
    if t >= KIRAI:
        S.stamp(cv, "キライ", 640, 92, 140, S.RED, t - KIRAI, -0.04, out=S.WHITE, ow=9)
        dt = t - KIRAI
        fx["shake"] = (14 * math.exp(-dt * 8) * (1 if int(dt * 24) % 2 else -1), 6 * math.exp(-dt * 8))
        fx["flash"] = 0.55 * math.exp(-dt * 14)
        fx["flash_color"] = (1.0, 0.9, 0.95)
        fx["ca"] = 1 + 5 * math.exp(-dt * 6)


_tex = None


def render(frame):
    global _tex
    if _tex is None:
        _tex = Tex()
    d = frame // TWOS
    t = T0 + d * TWOS / FPS
    cv = Canvas(seed=900 + d, boil=1.2)
    cv.rough = 1.0
    fx = dict(seed=d, grain_i=d, ca=1.2)
    background(cv, t)
    p, pulse = pose_for(t)
    start, _, _, mood = key_at(t)
    z = 1.3
    cx, cy = 640, 250
    if mood == "point":                      # наезд к пальцу
        k = min(1.0, (t - start) / 0.3)
        z = 1.3 + 0.12 * (1 - (1 - k) ** 3)
        cx -= 30 * k
    if mood == "kirai":
        k = min(1.0, (t - start) / 0.12)
        z = 1.25 - 0.05 * k
        cy = 250 - 70 * (1 - (1 - k) ** 2)
    if mood == "dai" and start > 4:          # замах: камера уже начинает подниматься
        cy = 220
    chara.draw(cv, p, S.root_xf(cx, cy, z, pulse, 0.03))
    if mood == "point":
        for j in range(10):                  # линии фокуса
            a = j * 0.628 + 0.3
            r0 = 520
            cv.stroke(np.array([(780 + r0 * math.cos(a), 220 + r0 * math.sin(a)),
                                (780 + (r0 + 400) * math.cos(a), 220 + (r0 + 400) * math.sin(a))]), 5, color=S.WHITE, key=3000 + j, alpha=0.8)
    overlays(cv, t, fx)
    if t - start < 0.09 and start in (DAI[0] - 0.02, 3.48):
        fx["glitch"] = 0.6
    S.subtitles(cv, t)
    return cv.finish(_tex, fx)


def work(d):
    p = os.path.join(TMP, f"t{d:05d}.png")
    if not os.path.exists(p):
        render(d * TWOS).save(p)
    return d


TMP = os.environ.get("FRAMES_DIR") or tempfile.mkdtemp(prefix="test_shot_")

if __name__ == "__main__":
    src, out = sys.argv[1], sys.argv[2]
    os.makedirs(TMP, exist_ok=True)
    n_draw = int((T1 - T0) * FPS) // TWOS
    with Pool(os.cpu_count()) as pool:
        list(pool.imap_unordered(work, range(n_draw)))
    with open(os.path.join(TMP, "list.txt"), "w") as f:
        for d in range(n_draw):
            f.write(f"file 't{d:05d}.png'\nduration {TWOS / FPS:.6f}\n")
        f.write(f"file 't{n_draw - 1:05d}.png'\n")
    dur = T1 - T0
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", os.path.join(TMP, "list.txt"),
                    "-ss", str(L.CLIP_T0 + T0), "-t", str(dur), "-i", src, "-map", "0:v", "-map", "1:a",
                    "-vf", f"fps={FPS},format=yuv420p", "-af", f"afade=t=in:d=0.15,afade=t=out:st={dur - 0.4}:d=0.4",
                    "-c:v", "libx264", "-crf", "20", "-preset", "slow", "-c:a", "aac", "-b:a", "192k", "-t", str(dur),
                    "-movflags", "+faststart", out], check=True)
    print("done", out)
