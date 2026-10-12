"""Отрезок 0:43–1:13 трека «ダイダイダイダイダイキライ» (雨良 Amala).

Время строк снято по смене субтитров в оригинальном клипе (секунды трека).
Перевод на английский — свой, вольный.
"""
import json
import os
import numpy as np

CLIP_T0 = 43.0        # начало отрезка в треке
CLIP_LEN = 30.0
HERE = os.path.dirname(os.path.abspath(__file__))

# (начало, конец, [(текст, фуригана|None)...], чтение для губ, перевод)
LINES = [
    (40.6, 44.2, [("赤信号", "あかしんごう"), ("にずっと", None), ("従順", "じゅうじゅん"), ("です。", None)],
     "あかしんごうにずっとじゅうじゅんです", "Always obedient to the red light."),
    (44.2, 46.4, [("大体", "だいたい"), ("　アナタのコト　アタシ", None)],
     "だいたいあなたのことあたし", "Basically, about you — me, I..."),
    (46.4, 48.2, [("ダイダイダイダイダイキライ", None)],
     "だいだいだいだいだいきらい", "hate-hate-hate-hate-HATE you"),
    (48.2, 49.8, [("OMG　", None), ("情", "なさ"), ("けない　", None), ("最早", "もはや")],
     "おーえむじーなさけないもはや", "OMG, how pathetic, at this point"),
    (49.8, 51.9, [("バイバイバイバイバイバイしたい", None)],
     "ばいばいばいばいばいばいしたい", "I wanna bye-bye-bye-bye-bye-bye"),
    (51.9, 53.7, [("脳内", "のうない"), ("も", None), ("予測不能", "よそくふのう"), ("　", None), ("故", "ゆえ"), ("に", None)],
     "のうないもよそくふのうゆえに", "My brain's unpredictable too, so"),
    (53.7, 55.5, [("解解解解解解", "かいかいかいかいかいかい"), ("は", None), ("無", "な"), ("い", None)],
     "かいかいかいかいかいかいはない", "there's no ans-ans-ans-answer"),
    (55.5, 57.5, [("どうだい？アタシのコト　アナタ", None)],
     "どうだいあたしのことあなた", "So how is it? Me — have you..."),
    (57.5, 59.9, [("ダイダイダイダイダイキライになった？", None)],
     "だいだいだいだいだいきらいになった", "...come to hate-hate-HATE me?"),
    (59.9, 62.3, [("あぁ　", None), ("百害", "ひゃくがい"), ("あって", None), ("一利", "いちり"), ("ナシ", None)],
     "あぁひゃくがいあっていちりなし", "Ah — all harm, not one bit of good"),
    (62.3, 63.3, [("Nothing Not see", None)], "なっしんぐなっとしー", "Nothing, not see"),
    (63.3, 67.1, [("もしタラレバ", None), ("並", "なら"), ("べてもニコチンだし", None)],
     "もしたらればならべてもにこちんだし", "Line up every what-if — still nicotine"),
    (67.1, 69.5, [("I miss youって", None), ("近付", "ちかづ"), ("いた", None), ("結末", "けつまつ"), ("は", None)],
     "あいみすゆーってちかづいたけつまつは", "Got close saying \"I miss you\", and in the end..."),
    (69.5, 70.6, [("凡", "ぼん"), ("ミス　", None), ("凡", "ぼん"), ("ミス", None)],
     "ぼんみすぼんみす", "a dumb slip, a dumb slip"),
    (70.6, 72.3, [("センキュー　", None), ("反面教師", "はんめんきょうし")],
     "せんきゅーはんめんきょうし", "Thank you, my perfect bad example"),
    (72.3, 75.6, [("後味", "あとあじ"), ("　", None), ("一生", "いっしょう"), ("消", "き"), ("えてくんないわ", None)],
     "あとあじいっしょうきえてくんないわ", "This aftertaste won't fade, not ever"),
]

VOWEL_OF = {}
for row, v in [("あかさたなはまやらわがざだばぱぁゃ", "a"), ("いきしちにひみりぎじぢびぴぃ", "i"),
               ("うくすつぬふむゆるぐずづぶぷぅゅ", "u"), ("えけせてねへめれげぜでべぺぇ", "e"),
               ("おこそとのほもよろをごぞどぼぽぉょ", "o")]:
    for c in row:
        VOWEL_OF[c] = v

_an = json.load(open(os.path.join(HERE, "analysis.json")))
BEATS = np.array(_an["beats"])          # время долей от начала отрезка
ENV = np.array(_an["env"])              # огибающая вокала (выделен REPET-SIM)
ENV_DT = _an["env_dt"]
DOWNBEAT0 = 3                           # индекс доли, где начинается припев (бочка)


def ct(abs_t):
    return abs_t - CLIP_T0


def vocal_env(t):
    i = int(t / ENV_DT)
    if i < 0 or i >= len(ENV):
        return 0.0
    w = ENV[max(0, i - 1):i + 2]
    return float(np.clip(w.mean(), 0, 1.2))


def _morae(reading):
    out = []
    for c in reading:
        if c in "ゃゅょぁぃぅぇぉ" and out:
            out[-1] = VOWEL_OF.get(c, out[-1])
        elif c in "っ":
            out.append("-")
        elif c in "ん":
            out.append("n")
        elif c == "ー" and out:
            out.append(out[-1])
        elif c in VOWEL_OF:
            out.append(VOWEL_OF[c])
    return out


def _build_mouth_track():
    """Гласные распределяются по «звучащим» кадрам каждой строки."""
    n = len(ENV)
    vow = [None] * n
    for a, b, _, reading, _ in LINES:
        i0, i1 = max(0, int(ct(a) / ENV_DT)), min(n, int(ct(b) / ENV_DT))
        if i1 <= i0:
            continue
        voiced = [i for i in range(i0, i1) if ENV[i] > 0.22]
        m = _morae(reading)
        if not voiced or not m:
            continue
        for k, i in enumerate(voiced):
            vow[i] = m[min(len(m) - 1, int(k / len(voiced) * len(m)))]
    return vow


MOUTH_TRACK = _build_mouth_track()


def mouth_at(t):
    i = int(t / ENV_DT)
    if i < 0 or i >= len(ENV):
        return None, 0.0
    e = vocal_env(t)
    v = MOUTH_TRACK[i]
    if v is None or e < 0.2:
        return None, 0.0
    if v in "-n":
        return "m", 0.2
    return v, min(1.0, (e - 0.15) * 1.6)


def line_at(t):
    at = t + CLIP_T0
    for k, L in enumerate(LINES):
        if L[0] <= at < L[1]:
            return k, L
    return None, None


def beat_phase(t):
    """(номер доли, фаза 0..1 внутри доли)."""
    i = int(np.searchsorted(BEATS, t, side="right")) - 1
    if i < 0:
        d = BEATS[1] - BEATS[0]
        return -1, (t - (BEATS[0] - d)) / d
    if i >= len(BEATS) - 1:
        d = BEATS[-1] - BEATS[-2]
        return i, (t - BEATS[-1]) / d % 1
    return i, (t - BEATS[i]) / (BEATS[i + 1] - BEATS[i])


def beat_pulse(t, decay=6.0, every=1, offset=0):
    """Импульс, затухающий после каждой доли."""
    i, ph = beat_phase(t)
    if (i - offset) % every:
        return 0.0
    d = (BEATS[1] - BEATS[0])
    return float(np.exp(-ph * d * decay))
