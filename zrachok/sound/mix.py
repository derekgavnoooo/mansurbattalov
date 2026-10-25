"""Сведение звука для «Зрачка» из присланных сэмплов.
Запуск: python3 mix.py <папка_с_сэмплами> <выходной.wav>
События (удары сердца, нырки в зрачок, помехи) считаются так же, как в index.html."""
import sys, subprocess, numpy as np

SR = 44100; DUR = 43.0; N = int(SR * DUR); FPS = 12
SRC, OUT = sys.argv[1], sys.argv[2]
rng = np.random.default_rng(7)

def load(name):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', f'{SRC}/{name}', '-f', 'f32le', '-ac', '2', '-ar', str(SR), '-'],
                         capture_output=True, check=True).stdout
    x = np.frombuffer(raw, np.float32).reshape(-1, 2).astype(np.float64)
    return x / (np.abs(x).max() + 1e-9)

# ---- те же функции, что в странице ----
def imul(a, b): return ((a & 0xffffffff) * (b & 0xffffffff)) & 0xffffffff
def h32(x, y):
    h = (imul(x, 374761393) + imul(y, 668265263)) & 0xffffffff
    h = imul(h ^ (h >> 13), 1274126177)
    return (h ^ (h >> 16)) / 4294967296
def sm(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t)
DK = [(0, 0), (4, 0), (11, 1), (18, 2), (24.5, 3), (30, 4), (34, 5), (38.3, 5.12)]
def depth(t):
    for (a, da), (b, db) in zip(DK, DK[1:]):
        if t < b:
            p = (t - a) / (b - a); p = p * p if a == 4 else p
            return da + (db - da) * p
    return DK[-1][1]
tension = lambda t: sm(4, 34, t)
CROSS = [11, 18, 24.5, 30, 34]
def tear(fr): return h32(fr, 31) < .02 + .12 * tension(fr / FPS) or any(abs(fr / FPS - c) < .13 for c in CROSS)
ph, beats = 0.0, []
for i in range(1, int(DUR * 100)):
    t = i / 100
    if t >= 38.8: break
    np_ = ph + (50 + 100 * tension(t)) / 60 / 100
    if int(np_) > int(ph): beats.append(t)
    ph = np_

# ---- обработка ----
def resample(x, rate):
    n = int(len(x) / rate); i = np.arange(n) * rate
    return np.stack([np.interp(i, np.arange(len(x)), x[:, c]) for c in (0, 1)], 1)
def stretch(x, f, g=.09):
    gl = int(g * SR); hop = gl // 4; n = int(len(x) * f); out = np.zeros((n + gl, 2)); w = np.hanning(gl)[:, None]
    for o in range(0, n, hop):
        s = int(o / f + rng.uniform(-.3, .3) * gl); s = min(max(s, 0), len(x) - gl - 1)
        out[o:o + gl] += x[s:s + gl] * w
    return out[:n] / 1.5
def filt(x, lo=None, hi=None, order=2):
    X = np.fft.rfft(x, axis=0); f = np.fft.rfftfreq(len(x), 1 / SR)[:, None]; m = np.ones_like(f)
    if hi: m /= np.sqrt(1 + (f / hi) ** (2 * order))
    if lo: m /= np.sqrt(1 + (lo / np.maximum(f, 1)) ** (2 * order))
    return np.fft.irfft(X * m, n=len(x), axis=0)
def env(x, a=.005, r=.05):
    n = len(x); e = np.ones(n); na, nr = int(a * SR), int(r * SR)
    if na: e[:na] = np.linspace(0, 1, na)
    if nr: e[-nr:] *= np.linspace(1, 0, nr)
    return x * e[:, None]

def add(a, b, gb=1.):
    n = max(len(a), len(b)); o = np.zeros((n, 2)); o[:len(a)] += a; o[:len(b)] += b * gb; return o

mix = np.zeros((N, 2)); verb = np.zeros((N, 2))
def put(x, t, g=1., pan=0., wet=.3):
    s = int(t * SR)
    if s >= N: return
    x = x[:N - s] * g * np.array([min(1, 1 - pan), min(1, 1 + pan)])
    mix[s:s + len(x)] += x; verb[s:s + len(x)] += x * wet

# сэмплы
heart = load('mixkit-human-single-heart-beat-490.wav')
choirA = load('mixkit-angelical-choir-654.wav'); choirH = load('mixkit-horror-choir-transition-660.wav')
whis = load('mixkit-male-conspiracy-voices-whispers-302.wav')
breath = load('mixkit-loud-creature-breathing-2231.wav')
splash = load('mixkit-gore-video-game-blood-splash-263.wav'); squish = load('mixkit-juicy-insect-fast-squish-324.wav')
stomp = load('mixkit-mech-stomp-squish-3047.wav'); bubble = load('mixkit-plastic-bubble-click-1124.wav')
bone = load('universfield-bone-crack-3-121580.mp3'); static = load('yourugor-tv-static-noise-291374.mp3')
scream = load('freesound_community-scream-90747.mp3'); storm = load('mixkit-storm-rumble-starting-2409.wav')

# 1. гул: растянутый хор, четыре копии по октавам, вечно ползёт вверх вместе с погружением
base = filt(stretch(np.concatenate([choirH, choirA[::-1], choirH[::-1], choirA]), 5.0), hi=3500)
base /= np.abs(base).max()
END = int(38.8 * SR); tt = np.arange(END) / SR
dep = np.interp(tt, np.arange(0, 43, .01), [depth(x) for x in np.arange(0, 43, .01)])
amp = (.18 + .5 * tension(tt)) * (1 - sm(38.4, 38.8, tt))
NV = 4; drone = np.zeros((END, 2))
for v in range(NV):
    pos = np.mod(dep + v, NV); rate = 2 ** (pos - 2.6); g = np.exp(-((pos - NV / 2) ** 2) / (2 * .75 ** 2))
    idx = np.mod(np.cumsum(rate) + v * 9e4, len(base) - 1)
    for c in (0, 1): drone[:, c] += np.interp(idx, np.arange(len(base)), base[:, c]) * g
drone = filt(drone, lo=40, hi=4000) * amp[:, None]
put(drone, 0, .55, wet=.5)
# низкий гром на старте и перед финалом
put(filt(resample(stretch(storm, 2.5), .55), hi=400), 0, .7, wet=.2)
put(env(filt(resample(stretch(storm, 2), .5), hi=500)[:int(7.8 * SR)], .5, 1.5), 31, .8, wet=.3)

# 2. сердце: каждый удар из той же таблицы, что и пульс плоти
hz = filt(resample(heart, .82), hi=900)
hz = add(hz, filt(resample(heart, .55), hi=200), .8)
for t in beats:
    put(env(hz, .001, .1), t, .55 + .5 * float(tension(t)), wet=.12)
put(env(hz, .001, .2), 42.6, 1.2, wet=.6)

# 3. нырок в зрачок: замедленный хлюп + удар
dive = add(filt(resample(splash, .55), hi=2500), filt(resample(stomp, .6), hi=1800), .8)
for c in CROSS: put(env(dive), c - .05, .9, wet=.45)
# моргания центрального глаза и закрытие в конце
lid = filt(resample(squish, .7), hi=3000)
for t in (13, 27): put(env(lid), t, .55, wet=.3)
put(env(filt(resample(squish, .45), hi=2000)), 38.25, .9, wet=.5)

# 4. мокрые щелчки маленьких глаз, их всё больше
wet_click = filt(resample(bubble, .45), hi=2500)
t = 9.0
while t < 38.6:
    put(env(wet_click, .001, .02), t, .25 + .3 * rng.random(), pan=rng.uniform(-.9, .9), wet=.4)
    t += rng.exponential(1.8 - 1.4 * float(tension(t)))

# 5. шёпот: замедлен, развёрнут, раздвоен по сторонам
w1 = filt(resample(whis, .78), lo=200, hi=3500); w2 = filt(resample(whis[::-1], .66), lo=150, hi=3000)
for k, s in enumerate(np.arange(14, 38.5, 3.6)):
    g = .12 + .3 * float(tension(s)); put(env(w1 if k % 2 else w2, .8, 1.), s, g, pan=-.7 if k % 2 else .7, wet=.6)
for s in (19.4, 35.3):  # все глаза смотрят на зрителя
    put(env(resample(whis, .9), .05, .5), s, .55, pan=0, wet=.35)
    put(env(resample(whis[::-1], .7), .3, .5), s + .4, .4, pan=.3, wet=.6)

# 6. дыхание существа, к финалу ближе и громче
br = filt(resample(breath, .62), hi=2500)
for s, g in ((7.5, .3), (16, .35), (22.5, .45), (28.5, .5), (33, .6), (35.6, .8), (37.0, .9)):
    put(env(br, .1, .3), s, g, wet=.35)

# 7. хруст кости на уровне со ртом
bn = filt(resample(bone, .7), hi=5000)
for s in (18.4, 19.9, 21.3, 22.6, 23.4, 24.2): put(env(bn, .001, .05), s, .6, pan=rng.uniform(-.6, .6), wet=.4)

# 8. помехи на рваных кадрах
st = filt(static, lo=600, hi=7000)
for fr in range(int(38.8 * FPS)):
    if tear(fr):
        L = int((.08 + .12 * h32(fr, 6)) * SR); o = int(h32(fr, 8) * (len(st) - L - 1))
        put(env(st[o:o + L], .002, .01), fr / FPS, .12 + .3 * float(tension(fr / FPS)), wet=.05)

# 9. вспышка глаза: крик замедлен и смешан с собой задом наперёд
sc = filt(resample(scream, .72), hi=5000)
put(env(sc[:int(.6 * SR)], .001, .2), 39.6, 1.0, wet=.5)
put(env(resample(scream[::-1], .5)[:int(.6 * SR)], .001, .2), 39.6, .6, wet=.6)

# ---- ревербератор: свёртка с затухающим шумом ----
n_ir = int(2.6 * SR); ir = rng.standard_normal((n_ir, 2)) * np.exp(-np.arange(n_ir) / (.45 * SR))[:, None]
ir = filt(ir, hi=3500)
L = 1 << int(np.ceil(np.log2(N + n_ir)))
rv = np.stack([np.fft.irfft(np.fft.rfft(verb[:, c], L) * np.fft.rfft(ir[:, c], L), L)[:N] for c in (0, 1)], 1)
out = mix + rv / (np.abs(rv).max() + 1e-9) * np.abs(mix).max() * .45
# тишина после закрытия глаза (кроме крика и последнего удара)
ts = np.arange(N) / SR
gate = np.ones(N)
gate[(ts >= 38.8) & (ts < 39.55)] = .02
tail = (ts >= 40.2) & (ts < 42.55); gate[tail] = .02 + .98 * (1 - sm(40.2, 40.7, ts[tail]))
out *= gate[:, None]
out = np.tanh(out / np.abs(out).max() * 1.6) * .89
pcm = (out * 32767).astype('<i2')
subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 's16le', '-ar', str(SR), '-ac', '2', '-i', '-', OUT], input=pcm.tobytes(), check=True)
print('beats', len(beats), 'tears', sum(tear(f) for f in range(int(38.8 * FPS))))
