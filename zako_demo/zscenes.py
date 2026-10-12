"""Монтаж «雑魚» (2:20.0–2:50.4): танец на общей сцене, затем три панели, загорающиеся по строкам."""
import json
import math
import os
import numpy as np
import rei  # noqa: F401
from flat import Canvas, hexc, SS, W, H
from rei import LINE
import poses

HERE = os.path.dirname(os.path.abspath(__file__))
AN = json.load(open(os.path.join(HERE, "analysis.json")))
T0, DUR = AN["t0"], AN["dur"]
BEATS = np.array(AN["beats"])
ENV = np.array(AN["env"])
FPS, TWOS = 24, 2
PINK = hexc("#fcc6dd")
BLACK = np.zeros(3)
PW = [0, 427, 853, 1280]           # границы трёх панелей

LINES = [
    (149.3, 151.5, [("いいこいいこしてあげようか", None)], "Want me to pat your head, good kid?"),
    (151.5, 153.3, [("目", "め"), ("が", None), ("合", "あ"), ("ったね　ドキドキしてるんだ", None)], "Our eyes met — your heart's pounding, huh"),
    (153.3, 155.5, [("何", "なに"), ("興奮", "こうふん"), ("してんの　", None), ("超", "ちょう"), ("キモいんだけど", None)], "Why so worked up? That's super gross"),
    (157.2, 158.7, [("なにそのくそざこ", None)], "What's with you, total small fry"),
    (158.7, 160.3, [("最底辺", "さいていへん"), ("よわよわゴミ", None)], "Rock-bottom, weak-weak trash"),
    (160.3, 162.7, [("けどほんとはすきすき", None)], "But honestly... I like-like you"),
    (164.3, 165.0, [("なにちょっと", None)], "Hey, wait—"),
    (165.0, 165.8, [("無視", "むし"), ("しないで", None)], "Don't ignore me"),
    (165.8, 166.7, [("謝", "あやま"), ("るからもう", None)], "I'll say sorry, so..."),
    (166.7, 167.5, [("置", "お"), ("いてかないで", None)], "Don't leave me behind"),
    (168.3, 169.8, [("嫌", "きら"), ("いになってない？", None)], "You don't hate me now... right?"),
    (169.8, 170.5, [("だよねw", None)], "Thought so lol"),
]

# (время, (левая, средняя, правая)); «*» — панели загораются по очереди на восьмых
PANELS = [
    (149.3, ("lying", None, None)),
    (151.5, (None, None, "chin")),
    (153.3, (None, "knees", None)),
    (155.5, ("*close_mouth",) * 3),
    (157.2, ("fists", None, None)),
    (158.7, (None, None, "shrug")),
    (160.3, (None, "cheeks", None)),
    (162.7, ("*close_cheeks",) * 3),
    (164.3, (None, "hand_mouth", None)),
    (165.0, ("fiddle", None, None)),
    (165.8, (None, None, "away")),
    (166.7, ("pinch", None, None)),
    (167.5, ("*close_pout", "*close_pout", None)),
    (168.3, (None, None, "worried")),
    (169.8, (None, "smug", None)),
]
DANCE_END = 149.3


def env_at(at):
    i = int((at - T0) / AN["env_dt"])
    return float(ENV[max(0, min(len(ENV) - 1, i))]) if 0 <= i < len(ENV) else 0.0


def mouth(at):
    e = env_at(at)
    return "open" if e > 0.5 else ("o" if e > 0.25 else None)


def beat_index(at):
    return int(np.searchsorted(BEATS, at, side="right")) - 1


def dance_frame(cv, at):
    b = beat_index(at)
    seq = [0, 1, 0, 3, 2, 1, 3, 0, 1, 2, 3, 0]
    k = seq[(max(b, 0) // 2) % len(seq)]
    eyes = "happy" if (b // 2) % 2 == 0 else "xx"
    m = mouth(at)
    if at < 145.87:
        x = [640, 470, 640, 810][(max(b, 0) // 2) % 4]
        poses.dance(cv, k, x, m=m, eyes=eyes)
    else:   # три Рэй — канон, как групповой танец в клипе
        for j, x in enumerate((213, 640, 1067)):
            kk = seq[((max(b, 0) // 2) + j) % len(seq)]
            poses.dance(cv, kk, x, m=m, eyes=eyes)


def panel_state(at):
    cur = None
    for t, slots in PANELS:
        if at >= t:
            cur = (t, slots)
    return cur


def render(frame):
    d = frame // TWOS
    at = T0 + d * TWOS / FPS
    cv = Canvas(seed=300 + d, boil=0.45, bg=PINK if at < DANCE_END else BLACK)
    if at < DANCE_END:
        dance_frame(cv, at)
    else:
        t, slots = panel_state(at)
        e8 = 0.2206
        for i, name in enumerate(slots):
            if name is None:
                continue
            if name.startswith("*"):
                if at < t + i * e8:
                    continue
                name = name[1:]
            pc = Canvas(seed=300 + d, boil=0.45, bg=PINK)
            poses.PANEL_POSES[name](pc, m=mouth(at))
            x0, x1 = PW[i] * SS, PW[i + 1] * SS
            cv.img[:, x0:x1] = pc.img[:, :x1 - x0]
    subtitles(cv, at)
    return cv.finish_clean()


def subtitles(cv, at):
    for a, b, segs, en in LINES:
        if a <= at < b:
            cv.ruby_line(segs, 640, 636, 32, color=(1, 1, 1), outline=LINE, ow=4, fam="zen-maru-gothic")
            cv.text(en, 640, 688, 20, "zen-maru-gothic", (1, 1, 1), LINE, 3, anchor="c")


if __name__ == "__main__":
    import sys
    tt = float(sys.argv[1])
    render(int(round((tt - T0) * FPS))).save(sys.argv[2])
