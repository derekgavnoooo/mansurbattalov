"""Кадр ролика в момент t. Анимация «по двойкам» (12 рисунков в секунду при 24 fps)."""
import math
import numpy as np
from ink import Canvas, Textures, catmull, ellipse, hexc, INK, PENCIL_BLUE, plen
import character as ch
import song

FPS = 24
TWOS = 2
TEAL_INK = hexc("#1d7f80")
GRAY = hexc("#8a8590")

_tex = None


def tex():
    global _tex
    if _tex is None:
        _tex = Textures(seed=3)
    return _tex


def smooth(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


BLINKS = [3.3, 9.6, 13.2, 17.9, 21.1, 25.4, 30.0]


def pose(t):
    p = dict(rot=0.0, dx=0.0, dy=0.0, breathe=0.0, tail_phase=0.0, tail_amp=0.0, blink=0,
             mouth=(None, 0), smile=1.0, glow=0.0)
    if t < song.COLOR_T[1]:
        return p
    p["breathe"] = 2.0 * math.sin(2 * math.pi * t / 3.2)
    p["tail_phase"] = 2 * math.pi * t / 2.6
    p["tail_amp"] = 6
    p["rot"] = 1.2 * math.sin(2 * math.pi * t / 4.0)
    if song.SONG_T0 <= t < song.END_T0:
        b = (t - song.SONG_T0) / song.BEAT
        p["rot"] = 3.6 * math.sin(math.pi * b / 2) + 0.8
        p["dy"] = -4 * abs(math.sin(math.pi * b))
        p["dx"] = 3 * math.sin(math.pi * b / 2)
        p["tail_phase"] = math.pi * b / 2 - 0.9
        p["tail_amp"] = 22
        p["glow"] = max(0.0, 1 - (b % 1) * 2.5)
        p["mouth"] = song.vocal_state(t)
    for bt in BLINKS:
        if 0 <= t - bt < 0.25:
            k = int((t - bt) / (1 / 12))
            p["blink"] = [1, 2, 1][min(k, 2)]
    # «счастливые глаза» на высоких нотах и в финале
    for n in song.NOTES:
        if n["midi"] >= song.C6 and n["start"] <= t < n["start"] + n["dur"]:
            p["blink"] = 3
    if t >= song.OUTRO_T0 + 0.2:
        p["blink"] = 3 if t < song.END_T0 + 1.6 else p["blink"]
        p["rot"] = 6.0 + 1.0 * math.sin(t * 2)
        p["mouth"] = (None, 0)
        p["smile"] = 2.0
    return p


# ---------------- расписание «рисования» линий ----------------

_sched = None


def schedule():
    """Порядок и время появления каждого штриха в фазе наброска."""
    global _sched
    if _sched is not None:
        return _sched
    parts = ch.build(pose(0))
    items = []
    for pi, part in enumerate(parts):
        for si, (pts, w, kw) in enumerate(part.strokes):
            items.append((part.order, pi, si, plen(pts)))
    items.sort(key=lambda x: (x[0], x[1], x[2]))
    t0, t1 = song.SKETCH_T
    cons = ch.construction(pose(0))
    cons_t = (t0, t0 + 0.9)
    dur = [0.05 + 0.0018 * L for *_, L in items]
    gap = 0.012
    total = sum(dur) + gap * len(items)
    k = (t1 - cons_t[1]) / total
    t = cons_t[1]
    sched = {}
    audio = []
    rng = np.random.default_rng(5)
    for (o, pi, si, L), d in zip(items, dur):
        d *= k
        sched[(pi, si)] = (t, t + d)
        audio.append((t, t + d, L))
        t += d + gap * k * rng.uniform(0.3, 1.7)
    cl = [plen(c) for c in cons]
    ct = cons_t[0]
    cs = []
    for L in cl:
        d = (cons_t[1] - cons_t[0]) * L / sum(cl)
        cs.append((ct, ct + d * 0.9))
        audio.append((ct, ct + d * 0.9, L))
        ct += d
    _sched = dict(strokes=sched, cons=cs, audio=audio)
    return _sched


def prog(t, a, b):
    if t >= b:
        return 1.0
    if t <= a:
        return 0.0
    return (t - a) / (b - a)


# ---------------- дудлы ----------------

def note_doodle(cv, x, y, s, rot, key, kind=0, alpha=0.9, fill=ch.C["hair"]):
    c, sn = math.cos(rot), math.sin(rot)
    T = lambda P: np.stack([x + s * (c * P[:, 0] - sn * P[:, 1]), y + s * (sn * P[:, 0] + c * P[:, 1])], 1)
    head = T(ellipse(0, 0, 7, 5, -0.4, 20))
    cv.fill(head, fill, alpha=0.9, key=key, edge=0.2, wobble=0.6)
    cv.stroke(head, 1.6, closed=True, key=key + 1, alpha=alpha)
    cv.stroke(T(np.array([[6, -2], [6.5, -20], [6, -36]])), 1.8, key=key + 2, alpha=alpha)
    if kind == 0:
        cv.stroke(T(catmull([(6, -36), (14, -28), (18, -18), (14, -10)])), 1.7, key=key + 3, alpha=alpha)
    else:
        head2 = T(ellipse(24, -6, 7, 5, -0.4, 20))
        cv.fill(head2, fill, alpha=0.9, key=key + 4, edge=0.2, wobble=0.6)
        cv.stroke(head2, 1.6, closed=True, key=key + 5, alpha=alpha)
        cv.stroke(T(np.array([[30, -8], [30.5, -26], [30, -42]])), 1.8, key=key + 6, alpha=alpha)
        cv.stroke(T(np.array([[6, -36], [18, -39], [30, -42]])), 3.2, key=key + 7, alpha=alpha, taper=(0.05, 0.05))


def sparkle(cv, x, y, s, key, color=INK):
    for a in (0, math.pi / 2):
        P = np.array([[x - s * math.cos(a), y - s * math.sin(a)], [x + s * math.cos(a), y + s * math.sin(a)]])
        cv.stroke(P, 1.5, color=color, key=key + int(a * 10), overshoot=0, taper=(0.4, 0.4))


# ---------------- кадр ----------------

def render(frame):
    t = (frame // TWOS) * TWOS / FPS   # время рисунка (по двойкам)
    drawing = frame // TWOS
    cv = Canvas(tex(), seed=1000 + drawing)
    P = pose(t)
    parts = ch.build(P)
    sch = schedule()
    sketching = t < song.SKETCH_T[1]
    c0, c1 = song.COLOR_T
    color_p = prog(t, c0, c1)

    for pi, part in enumerate(parts):
        for occ in part.occ:
            cv.occlude([occ])
        if color_p > 0:
            # раскраска идёт по частям, со сдвигом
            rv = smooth(color_p * 1.6 - (part.order - 1) * 0.06)
            for k, (poly, color, kw) in enumerate(part.fills):
                kw = dict(kw)
                key = kw.pop("key", k) + 17 * pi
                cv.fill(poly, color, reveal=rv if rv < 1 else 1.0, key=key, **kw)
            for k, (poly, a, g, kw) in enumerate(part.hatches):
                cv.hatch(poly, a, g, key=900 + pi * 10 + k, prog=rv, **kw)
        for si, (pts, w, kw) in enumerate(part.strokes):
            a, b = sch["strokes"].get((pi, si), (0, 0))
            pr = prog(t, a, b)
            if pr > 0:
                cv.stroke(pts, w, prog=pr, key=pi * 1000 + si, **kw)
        if hasattr(part, "tattoo") and color_p > 0.6:
            (tx, ty), s = part.tattoo
            cv.text(s, tx, ty, 30, color=ch.C["red"], alpha=0.85 * smooth((color_p - 0.6) / 0.4), key=77, rot=0.12)

    # недотёртые построения поверх всего
    cons = ch.construction(P if not sketching else pose(0))
    ca = 0.42 if sketching else 0.42 - 0.24 * smooth(color_p)
    for k, (L, (a, b)) in enumerate(zip(cons, sch["cons"])):
        pr = prog(t, a, b)
        if pr > 0:
            cv.stroke(L, 1.1, color=PENCIL_BLUE, alpha=ca, prog=pr, grain=0.9, key=50000 + k, sketch=0.6,
                      taper=(0.3, 0.3))

    # ---------- текст ----------
    tp = prog(t, *song.TITLE_T)
    cv.text("Нотка", 810, 150, 112, prog=tp, key=1)
    if tp >= 1:
        sp = prog(t, song.TITLE_T[1], song.TITLE_T[1] + 0.6)
        cv.text("вокал-демо #1", 836, 222, 46, prog=sp, key=2, color=hexc("#3b3640"))
        und = catmull([(812, 168), (880, 172), (960, 166), (1040, 171), (1095, 164)])
        cv.stroke(und, 2.4, prog=prog(t, 1.9, 2.3), color=ch.C["hair_sh"], key=3, alpha=0.8)
        if t > 2.2:
            note_doodle(cv, 1135, 108, 1.4, 0.15, key=60, kind=1)
            sparkle(cv, 1180, 70, 9, key=70)
            sparkle(cv, 790, 82, 6, key=71)
    # подпись-дата в углу
    cv.text("12.10 — рисую от руки, не судите строго :)", 860, 700, 28, color=GRAY, alpha=0.75, key=5,
            prog=prog(t, 2.0, 3.0))

    # ---------- караоке-субтитры ----------
    if song.VOX_T0 - 0.8 <= t < song.OUTRO_T0 + 0.3:
        phrase_t = [min(n["start"] for n in song.NOTES if n["phrase"] == k) for k in range(4)]
        ph = max([k for k in range(4) if t >= phrase_t[k] - 0.8] or [0])
        words = song.PHRASE_WORDS[ph]
        syl_notes = [n for n in song.NOTES if n["phrase"] == ph]
        text, cols, si = "", [], 0
        for wi, wd in enumerate(words):
            for sy in wd:
                n = syl_notes[si]
                sung = t >= n["start"]
                for chh in sy:
                    text += chh
                    cols.append(TEAL_INK if sung else GRAY)
                si += 1
            text += " "
            cols.append(INK)
        appear = prog(t, phrase_t[ph] - 0.8, phrase_t[ph] - 0.35)
        cv.text(text.strip(), 735, 560, 50, colors=cols, prog=appear, key=200 + ph * 7)
        if ph < 3:
            nxt = " ".join("".join(w) for w in song.PHRASE_WORDS[ph + 1])
            cv.text(nxt, 760, 612, 34, color=GRAY, alpha=0.38, key=300 + ph * 7)

    # ---------- ноты, вылетающие из гарнитуры ----------
    if song.VOX_T0 <= t < song.END_T0 + 1:
        for i, n in enumerate(song.NOTES):
            if i % 3 != 0 and n["dur"] < 0.7:
                continue
            age = t - n["start"]
            if 0 <= age < 2.2:
                r = np.random.default_rng(i)
                x = 560 + r.uniform(0, 70) + age * r.uniform(30, 60) + 8 * math.sin(age * 4 + i)
                y = 250 - age * r.uniform(55, 85) + r.uniform(-30, 30)
                s = 1.0 + 0.25 * r.random()
                fade = 1 - smooth((age - 1.6) / 0.6)
                col = [ch.C["hair"], ch.C["pink"], hexc("#f3c64a")][i % 3]
                note_doodle(cv, x, y, s, r.normal(0, 0.25) + 0.2 * math.sin(age * 3), key=4000 + i * 10,
                            kind=i % 2, alpha=0.9 * fade, fill=col)
        # «звуковые» дуги у губ на громких нотах
        v, op = P["mouth"]
        if v and op > 0.5 and v != "m":
            for k in range(2):
                R = 60 + 18 * k
                arc = np.array([[ch.X0 + 40 + R * math.cos(a), ch.Y0 + 112 + R * math.sin(a)]
                                for a in np.linspace(-0.45, 0.45, 12)])
                cv.stroke(arc, 1.3, alpha=0.5, key=7000 + k, taper=(0.4, 0.4))

    # ---------- финал ----------
    if t >= song.END_T0 - 0.6:
        ep = prog(t, song.END_T0 - 0.6, song.END_T0 + 0.9)
        cv.text("спасибо, что послушали!", 730, 470, 56, prog=ep, key=400, color=TEAL_INK)
        if t > song.END_T0 + 1.0:
            cv.text("— Нотка", 1010, 515, 40, key=401, prog=prog(t, song.END_T0 + 1.0, song.END_T0 + 1.5))
            heart = catmull([(0, 10), (-14, -2), (-12, -14), (-3, -14), (0, -6), (3, -14), (12, -14), (14, -2), (0, 10)])
            hp = prog(t, song.END_T0 + 1.5, song.END_T0 + 2.0)
            cv.stroke(heart * 1.3 + np.array([1170, 500]), 2.2, prog=hp, color=ch.C["red"], key=402)

    # скотч по углам листа
    cv.tape(30, 22, 140, 44, -0.65, key=11)
    cv.tape(1258, 30, 130, 42, 0.55, key=12)
    return cv.result()


if __name__ == "__main__":
    import sys
    t = float(sys.argv[1])
    out = sys.argv[2]
    render(int(round(t * FPS))).save(out)
