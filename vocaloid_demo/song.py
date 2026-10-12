"""Песня: таймлайн, мелодия, слова и синтез «вокалоидного» голоса + минус.

Голос — аддитивный формантный синтез (как ранние синтезаторы пения):
гармоники основного тона, огибающая которых задаётся формантами гласных.
"""
import numpy as np

SR = 44100
BPM = 120
BEAT = 60.0 / BPM
EIGHTH = BEAT / 2
BAR = BEAT * 4

# --- таймлайн ролика (секунды) ---
TITLE_T = (0.2, 1.2)      # пишется заголовок
SKETCH_T = (0.9, 7.0)     # рисуются линии
COLOR_T = (6.9, 8.6)      # раскраска
SONG_T0 = 8.5             # такт вступления
VOX_T0 = SONG_T0 + BAR    # начало вокала
VOX_BARS = 8
OUTRO_T0 = VOX_T0 + VOX_BARS * BAR
END_T0 = OUTRO_T0 + BAR
TOTAL = END_T0 + 3.0

C5, D5, E5, F5, G5, A5, B5, C6 = 72, 74, 76, 77, 79, 81, 83, 84
R = None  # пауза

# (слог, нота, длительность в восьмых); каждая фраза = 2 такта = 16 восьмых
PHRASES = [
    [("я", E5, 1), ("на", E5, 1), ("ри", D5, 1), ("со", E5, 1), ("ва", G5, 2),
     ("на", E5, 2), ("ка", D5, 1), ("ран", C5, 1), ("да", D5, 1), ("шом", E5, 3), (R, R, 2)],
    [("ли", C5, 1), ("ни", D5, 1), ("и", E5, 2), ("дро", G5, 1), ("жат,", A5, 3),
     ("но", G5, 1), ("я", E5, 1), ("по", D5, 2), ("ю", C5, 3), (R, R, 1)],
    [("ля", A5, 1), ("ля", G5, 1), ("ля", E5, 2), ("ля", A5, 1), ("ля", G5, 1), ("ля", E5, 2),
     ("ля", D5, 1), ("ля", E5, 1), ("ля", G5, 1), ("ля", A5, 1), ("ля!", C6, 3), (R, R, 1)],
    [("слы", A5, 1), ("шишь", G5, 1), ("мой", E5, 2), ("го", D5, 1), ("лос?", E5, 2), (R, R, 1),
     ("э", G5, 1), ("то", A5, 1), ("де", G5, 2), ("мо!", C5, 4)],
]
# как фразы пишутся в субтитрах (слоги склеиваются по словам)
PHRASE_WORDS = [
    [["я"], ["на", "ри", "со", "ва", "на"], ["ка", "ран", "да", "шом"]],
    [["ли", "ни", "и"], ["дро", "жат,"], ["но"], ["я"], ["по", "ю"]],
    [["ля", "ля", "ля,"], ["ля", "ля", "ля,"], ["ля", "ля", "ля", "ля", "ля!"]],
    [["слы", "шишь"], ["мой"], ["го", "лос?"], ["э", "то"], ["де", "мо!"]],
]
CHORDS = ["C", "Am", "F", "G", "C", "Am", "F", "C"]  # по такту вокала
CHORD_NOTES = {"C": [60, 64, 67], "Am": [57, 60, 64], "F": [53, 57, 60], "G": [55, 59, 62]}
CHORD_ROOT = {"C": 36, "Am": 33, "F": 29, "G": 31}

VOWELS = {"а": "a", "я": "a", "о": "o", "ё": "o", "у": "u", "ю": "u",
          "и": "i", "ы": "y", "э": "e", "е": "e"}
IOTATED = set("яёюе")
# форманты сопрано (F1..F4, Гц)
FORM = {
    "a": (880, 1300, 2900, 3900), "o": (560, 950, 2800, 3800), "u": (380, 820, 2700, 3700),
    "i": (330, 2700, 3300, 4100), "e": (560, 2150, 2950, 3900), "y": (380, 1650, 2750, 3800),
    "l": (360, 1050, 2600, 3700), "n": (280, 1500, 2600, 3700), "m": (260, 1100, 2500, 3600),
    "r": (520, 1300, 2000, 3600), "j": (300, 2500, 3200, 4000), "v": (300, 1200, 2400, 3600),
    "zh": (350, 1800, 2400, 3500), "d": (400, 1700, 2600, 3700),
}
VOICED_CONS = {"л": "l", "н": "n", "м": "m", "р": "r", "й": "j", "в": "v", "ж": "zh",
               "з": "zh", "д": "d", "б": "m", "г": "d"}
NOISE_CONS = {"с": "s", "ш": "sh", "щ": "sh", "ч": "sh", "ф": "f", "х": "h",
              "т": "t", "к": "k", "п": "p", "ж": "sh", "з": "s", "д": "t", "б": "p", "г": "k"}


def midi_hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def parse_syl(s):
    s = s.strip(",.!?").lower()
    i = next(k for k, c in enumerate(s) if c in VOWELS)
    onset, v, coda = s[:i], s[i], s[i + 1:].replace("ь", "")
    return onset, VOWELS[v], v in IOTATED, coda


def build_notes():
    notes, t = [], VOX_T0
    for pi, ph in enumerate(PHRASES):
        k = 0
        for syl, m, d in ph:
            dur = d * EIGHTH
            if syl is not None:
                onset, vow, iot, coda = parse_syl(syl)
                notes.append(dict(syl=syl, midi=m, start=t, dur=dur, onset=onset, vowel=vow,
                                  iot=iot, coda=coda, phrase=pi, idx=k))
                k += 1
            t += dur
    for a, b in zip(notes, notes[1:]):
        a["legato"] = abs(b["start"] - (a["start"] + a["dur"])) < 1e-6
    notes[-1]["legato"] = False
    return notes


NOTES = build_notes()


def note_end(n):
    return n["start"] + n["dur"] - (0.02 if n["legato"] else 0.07)


MOUTH_OPEN = {"a": 1.0, "o": 0.8, "u": 0.45, "i": 0.45, "e": 0.7, "y": 0.5}


def vocal_state(t):
    """Форма рта в момент t: (гласная | 'm' | None, раскрытие 0..1)."""
    for n in NOTES:
        s, e = n["start"], note_end(n)
        if s - 0.07 <= t < s:
            return ("m" if n["onset"][-1:] in "мбп" else n["vowel"]), 0.25 if n["onset"] else 0.0
        if s <= t < e:
            if n["coda"] and t > e - 0.09:
                return ("m" if n["coda"][-1] in "мбп" else n["vowel"]), 0.3
            att = min(1.0, (t - s) / 0.05)
            return n["vowel"], MOUTH_OPEN[n["vowel"]] * att
    return None, 0.0


# ======================= синтез =======================

def _smooth(x, ms):
    k = max(1, int(SR * ms / 1000))
    ker = np.hanning(k + 2)[1:-1]
    ker /= ker.sum()
    if x.ndim == 1:
        return np.convolve(x, ker, mode="same")
    return np.stack([np.convolve(x[:, i], ker, mode="same") for i in range(x.shape[1])], 1)


def synth_voice(n_samples, rng):
    from scipy.signal import butter, sosfilt
    f0 = np.full(n_samples, midi_hz(NOTES[0]["midi"]))
    F = np.tile(np.array(FORM["a"], float), (n_samples, 1))
    amp = np.zeros(n_samples)
    noise_env = {k: np.zeros(n_samples) for k in ["s", "sh", "f", "h", "t", "k", "p"]}
    vib = np.zeros(n_samples)
    S = lambda x: int(round(x * SR))

    def put_cons(c, a, b, after_vowel=False):
        a, b = S(a), S(b)
        if c in VOICED_CONS:
            F[a:b] = FORM[VOICED_CONS[c]]
            amp[a:b] = {"л": 0.55, "н": 0.4, "м": 0.35, "р": 0.6, "й": 0.7, "в": 0.45}.get(c, 0.35)
            if c == "р":  # одноударное «р»
                tt = np.arange(b - a) / SR
                amp[a:b] *= 0.6 + 0.4 * np.cos(2 * np.pi * 28 * tt)
        else:
            amp[a:b] = 0.0
        if c in NOISE_CONS:
            kind = NOISE_CONS[c]
            if kind in "tkp":  # смычка + взрыв
                bl = min(b - a, S(0.022))
                noise_env[kind][b - bl:b] = np.linspace(1, 0.2, bl) ** 2
            else:
                noise_env[kind][a:b] = 1.0

    for i, n in enumerate(NOTES):
        s, e = n["start"], note_end(n)
        hz = midi_hz(n["midi"])
        a = S(s - 0.075)
        f0[a:S(e) + S(0.06)] = hz
        # согласные начала слога
        on = n["onset"]
        if on:
            seg = 0.075 / len(on)
            for j, c in enumerate(on):
                put_cons(c, s - 0.075 + j * seg, s - 0.075 + (j + 1) * seg)
        # гласная
        cend = e - (0.08 if n["coda"] else 0.0)
        F[S(s):S(cend)] = FORM[n["vowel"]]
        if n["iot"]:
            F[S(s):S(s + 0.045)] = FORM["j"]
        env = np.ones(S(cend) - S(s))
        rel = min(len(env), S(0.06))
        if not n["legato"] and not n["coda"]:
            env[-rel:] *= np.linspace(1, 0, rel) ** 1.5
        amp[S(s):S(cend)] = env * (0.9 + 0.1 * (n["midi"] - 72) / 12)
        if n["coda"]:
            seg = 0.08 / len(n["coda"])
            for j, c in enumerate(n["coda"]):
                put_cons(c, cend + j * seg, cend + (j + 1) * seg)
            amp[S(e) - S(0.02):S(e)] *= np.linspace(1, 0, S(e) - (S(e) - S(0.02)))
        # вибрато нарастает на длинных нотах
        if n["dur"] > 0.4:
            va, vb = S(s + 0.22), S(e)
            vib[va:vb] = np.linspace(0, 1, vb - va) ** 0.7

    # портаменто, плавные переходы формант
    f0 = np.exp(_smooth(np.log(f0), 45))
    F = _smooth(F, 22)
    amp = _smooth(amp, 12)
    tt = np.arange(n_samples) / SR
    drift = _smooth(rng.standard_normal(n_samples), 120) * 60
    f0 = f0 * 2 ** ((38 * vib * np.sin(2 * np.pi * 5.7 * tt) + drift * 4) / 1200)

    # аддитивный формантный синтез
    out = np.zeros(n_samples)
    G = np.array([1.0, 0.55, 0.32, 0.18])
    B = np.array([110.0, 130.0, 190.0, 280.0])
    phase = 2 * np.pi * np.cumsum(f0) / SR
    chunk = SR
    for c0 in range(0, n_samples, chunk):
        c1 = min(n_samples, c0 + chunk)
        if amp[c0:c1].max() < 1e-4:
            continue
        ff, FF, ph = f0[c0:c1], F[c0:c1], phase[c0:c1]
        acc = np.zeros(c1 - c0)
        for h in range(1, 40):
            fh = h * ff
            if fh.min() > 10000:
                break
            gain = 0.015 + (G / (1 + ((fh[:, None] - FF) / (B / 2)) ** 2)).sum(1)
            gain *= h ** -0.55 * (fh < 10000)
            acc += gain * np.sin(h * ph + h * 0.37)
        out[c0:c1] = acc * amp[c0:c1]
    # придыхание
    breath = sosfilt(butter(2, [700, 5000], "band", fs=SR, output="sos"), rng.standard_normal(n_samples))
    out += breath * amp * 0.035
    # шумные согласные
    bands = {"s": (4500, 10000, 0.22), "sh": (1800, 5500, 0.28), "f": (900, 7000, 0.07),
             "h": (700, 3000, 0.12), "t": (2800, 9000, 0.5), "k": (1300, 4000, 0.45), "p": (300, 2500, 0.35)}
    for k, (lo, hi, g) in bands.items():
        ev = _smooth(noise_env[k], 6)
        if ev.max() == 0:
            continue
        nz = sosfilt(butter(3, [lo, hi], "band", fs=SR, output="sos"), rng.standard_normal(n_samples))
        out += nz * ev * g
    return out / (np.abs(out).max() + 1e-9)


def synth_backing(n_samples, rng):
    from scipy.signal import butter, sosfilt
    out = np.zeros(n_samples)
    tt = np.arange(n_samples) / SR

    def add(sig, t0, gain=1.0):
        a = int(t0 * SR)
        if a >= n_samples:
            return
        b = min(n_samples, a + len(sig))
        out[a:b] += sig[: b - a] * gain

    def pluck(m, dur, bright=1.0):
        n = int(dur * SR)
        t = np.arange(n) / SR
        f = midi_hz(m)
        sig = sum((0.6 ** k) * np.sin(2 * np.pi * f * (k + 1) * t + k) for k in range(5))
        sig += 0.2 * bright * np.sign(np.sin(2 * np.pi * f * t))  # чуть «чиптюна»
        return sig * np.exp(-t * 7) * np.minimum(1, t / 0.004)

    def kick():
        t = np.arange(int(0.35 * SR)) / SR
        f = 45 + 110 * np.exp(-t * 28)
        return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 9)

    hp = butter(2, 6000, "high", fs=SR, output="sos")
    bp = butter(2, [900, 6000], "band", fs=SR, output="sos")

    def hat(o=False):
        n = int((0.18 if o else 0.05) * SR)
        return sosfilt(hp, rng.standard_normal(n)) * np.exp(-np.arange(n) / SR * (14 if o else 70))

    def snare():
        n = int(0.22 * SR)
        t = np.arange(n) / SR
        return (sosfilt(bp, rng.standard_normal(n)) * 0.8 + 0.5 * np.sin(2 * np.pi * 190 * t)) * np.exp(-t * 18)

    bars = [("C", SONG_T0)] + [(c, VOX_T0 + i * BAR) for i, c in enumerate(CHORDS)] + [("C", OUTRO_T0)]
    for bi, (ch, t0) in enumerate(bars):
        notes = CHORD_NOTES[ch]
        arp = [notes[0] + 12, notes[1] + 12, notes[2] + 12, notes[1] + 24, notes[2] + 12, notes[1] + 12, notes[0] + 24, notes[2] + 12]
        last = bi == len(bars) - 1
        for k in range(8):
            if last and k > 0:
                break
            add(pluck(arp[k], 0.45), t0 + k * EIGHTH, 0.13)
            add(pluck(CHORD_ROOT[ch] + (12 if k % 2 else 0), 0.3, 0.3), t0 + k * EIGHTH, 0.22)
        # пэд
        n = int((BAR if not last else BAR * 1.5) * SR)
        t = np.arange(n) / SR
        pad = sum(np.sin(2 * np.pi * midi_hz(m + 12) * t * (1 + d)) for m in notes for d in (-0.003, 0.003))
        pad *= np.minimum(1, t / 0.25) * np.minimum(1, (n / SR - t) / 0.3)
        add(pad, t0, 0.035)
        # барабаны
        for b in range(4):
            tb = t0 + b * BEAT
            if last and b > 0:
                break
            if b in (0, 2) or (bi == 0 and b == 3):
                add(kick(), tb, 0.55)
            if b in (1, 3) and bi > 0:
                add(snare(), tb, 0.3)
            if bi > 0:
                add(hat(), tb + EIGHTH, 0.12)
                add(hat(b == 3), tb, 0.06)
        if last:
            add(kick(), t0, 0.5)
    # тихая «музыкальная шкатулка» во время рисования
    box = [76, 79, 84, 83, 79, 76, 74, 76, 72, 74, 76, 79]
    for i, m in enumerate(box):
        add(pluck(m + 12, 1.2, 0.0), 1.0 + i * 0.52, 0.05)
    return out


def synth_pencil(n_samples, strokes, rng):
    """Шорох карандаша: strokes = [(t0, t1, length_px)]."""
    from scipy.signal import butter, sosfilt
    nz = rng.standard_normal(n_samples)
    tex = sosfilt(butter(2, [1800, 7500], "band", fs=SR, output="sos"), nz)
    grit = (rng.random(n_samples) < 0.004) * rng.standard_normal(n_samples) * 3
    tex = tex + sosfilt(butter(2, 2500, "high", fs=SR, output="sos"), grit)
    env = np.zeros(n_samples)
    for t0, t1, L in strokes:
        a, b = int(t0 * SR), int(t1 * SR)
        if b <= a + 8:
            b = a + 8
        n = b - a
        x = np.linspace(0, 1, n)
        speed = np.sin(np.pi * x) ** 0.6 * (0.6 + 0.4 * rng.random())
        env[a:b] = np.maximum(env[a:b], speed[: len(env[a:b])])
    env = _smooth(env, 5)
    wobble = 0.7 + 0.3 * _smooth(rng.standard_normal(n_samples), 30) * 8
    return tex * env * np.clip(wobble, 0.2, 1.4)


def render_audio(path, pencil_strokes, chime_t=COLOR_T[0]):
    from scipy.io import wavfile
    from scipy.signal import fftconvolve
    rng = np.random.default_rng(7)
    n = int(TOTAL * SR)
    voice = synth_voice(n, rng)
    back = synth_backing(n, rng)
    pencil = synth_pencil(n, pencil_strokes, rng)
    # «звёздочки» при раскраске
    t = np.arange(int(1.5 * SR)) / SR
    chime = np.zeros(n)
    for k, m in enumerate([84, 88, 91, 96]):
        a = int((chime_t + k * 0.12) * SR)
        s = np.sin(2 * np.pi * midi_hz(m) * t) * np.exp(-t * 4)
        chime[a:a + len(s)] += s[: n - a] * 0.08
    # реверб: свёртка с затухающим шумом
    ir_t = np.arange(int(1.1 * SR)) / SR
    ir = rng.standard_normal(len(ir_t)) * np.exp(-ir_t * 5.5)
    ir[0] = 0
    ir /= np.abs(ir).sum() / 6

    def verb(x, wet):
        return x + wet * fftconvolve(x, ir)[:n]

    mix = verb(voice * 0.42, 0.35) + verb(back * 0.9, 0.2) + pencil * 0.16 + verb(chime, 0.5)
    mix = np.tanh(mix * 1.1) / np.tanh(1.1)
    fade = int(1.0 * SR)
    mix[-fade:] *= np.linspace(1, 0, fade)
    mix /= np.abs(mix).max() / 0.89
    st = np.stack([mix, mix], 1)
    # лёгкая стереобаза
    d = int(0.012 * SR)
    st[d:, 1] = 0.6 * st[d:, 1] + 0.4 * mix[:-d]
    wavfile.write(path, SR, (st * 32767).astype(np.int16))
