"""Звук для «Глубины» из присланных сэмплов.
Запуск: python3 mix.py <папка_с_сэмплами> <выходной.wav>
Время событий совпадает с index.html."""
import sys, subprocess, numpy as np

SR = 44100; DUR = 45.0; N = int(SR * DUR)
SRC, OUT = sys.argv[1], sys.argv[2]
rng = np.random.default_rng(11)

def load(name):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', f'{SRC}/{name}', '-f', 'f32le', '-ac', '2', '-ar', str(SR), '-'],
                         capture_output=True, check=True).stdout
    x = np.frombuffer(raw, np.float32).reshape(-1, 2).astype(np.float64)
    return x / (np.abs(x).max() + 1e-9)
def imul(a, b): return ((a & 0xffffffff) * (b & 0xffffffff)) & 0xffffffff
def h32(x, y):
    h = (imul(x, 374761393) + imul(y, 668265263)) & 0xffffffff
    h = imul(h ^ (h >> 13), 1274126177)
    return (h ^ (h >> 16)) / 4294967296
def sm(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t)
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
    n = len(x); e = np.ones(n); na, nr = min(n, int(a * SR)), min(n, int(r * SR))
    if na: e[:na] = np.linspace(0, 1, na)
    if nr: e[-nr:] *= np.linspace(1, 0, nr)
    return x * e[:, None]
def add(a, b, gb=1.):
    n = max(len(a), len(b)); o = np.zeros((n, 2)); o[:len(a)] += a; o[:len(b)] += b * gb; return o

ts = np.arange(N) / SR
above = np.zeros((N, 2)); under = np.zeros((N, 2)); verb = np.zeros((N, 2))
def put(bus, x, t, g=1., pan=0., wet=.3):
    s = int(t * SR)
    if s >= N: return
    x = x[:N - s] * g * np.array([min(1, 1 - pan), min(1, 1 + pan)])
    bus[s:s + len(x)] += x; verb[s:s + len(x)] += x * wet

heart = load('mixkit-human-single-heart-beat-490.wav')
choirA = load('mixkit-angelical-choir-654.wav'); choirH = load('mixkit-horror-choir-transition-660.wav')
breath = load('mixkit-loud-creature-breathing-2231.wav')
splash = load('mixkit-gore-video-game-blood-splash-263.wav'); squish = load('mixkit-juicy-insect-fast-squish-324.wav')
stomp = load('mixkit-mech-stomp-squish-3047.wav'); bubble = load('mixkit-plastic-bubble-click-1124.wav')
bone = load('universfield-bone-crack-3-121580.mp3'); scream = load('freesound_community-scream-90747.mp3')
storm = load('mixkit-storm-rumble-starting-2409.wav')

# ---------- над водой (0–21.6) ----------
# море: накаты волн из шума с медленными подъёмами
nz = rng.standard_normal((N, 2))
sea = filt(nz, hi=500) * 3
swell = .55 + .45 * np.sin(ts * TAU / 4.3 if (TAU := 2 * np.pi) else 0) ** 2
hiss = filt(nz[::-1], lo=1500, hi=6000) * .5 * np.clip(np.sin(ts * TAU / 4.3 + .6), 0, 1)[:, None] ** 3
sea = (sea * swell[:, None] + hiss) * (.55 + .4 * sm(13, 19, ts))[:, None]
put(above, sea, 0, .35, wet=.1)
# далёкий ревун маяка: растянутый и опущенный хор
horn = filt(resample(stretch(choirH[:int(2.5 * SR)], 1.6), .35), hi=500)
for t in (2.5, 10, 17.4): put(above, env(horn, .6, 1.2), t, .5, pan=-.5, wet=.8)
# скрип лодки: хруст кости, сильно замедленный
creak = filt(resample(bone, .32), lo=150, hi=2500)
for t in (1.2, 5.4, 8.8, 13.1, 15.6, 17.0, 18.7, 19.6): put(above, env(creak, .01, .2), t, .35 + .03 * t, pan=rng.uniform(-.4, .4), wet=.3)
# всплески там, где что-то высовывается из воды
OBJ = [(3.5, 46, 14), (6.5, 30, -11), (9, 24, 6), (9.6, 24, 8.5), (11.8, 13, -4)] + \
      [(14.6 + k * .35, 8.5, -14 + k * 3) for k in range(9)] + [(18.3, 3.2, 1.5)]
lap = filt(resample(squish, .5), hi=1500)
for t0, Z, X in OBJ: put(above, env(lap, .01, .3), t0 + .4, min(1, 4 / Z), pan=float(np.clip(X / 15, -.8, .8)), wet=.5)
# огромная тень под водой: глубокий выдох и гром
put(above, env(filt(resample(stretch(breath, 2.2), .3), hi=600), .5, 1), 14, 1.0, wet=.4)
put(above, env(filt(resample(storm, .45), hi=300), .8, 1.5), 14.2, .9, wet=.2)
# щупальце у борта, переворот
put(above, env(add(filt(resample(splash, .6), hi=3000), filt(resample(stomp, .55), hi=1200), .9)), 18.4, .9, wet=.4)
put(above, env(add(filt(resample(splash, .45), hi=2500), filt(resample(storm, .6), hi=800)[:int(2 * SR)], .8), .001, .3), 20.9, 1.2, wet=.5)
above[ts >= 21.6] = 0

# ---------- под водой (21.6–39.4) ----------
# гул: хор растянут, опущен и задушен водой
drone = filt(resample(stretch(np.concatenate([choirH, choirA[::-1], choirH]), 3.2), .45), lo=30, hi=700)
dr = np.zeros((N, 2)); s0 = int(21.6 * SR); dr[s0:s0 + len(drone)] = drone[:N - s0]
dr *= ((.25 + .6 * sm(22, 38, ts)) * (ts < 39.4))[:, None]
put(under, dr, 0, .6, wet=.4)
# первый провал под воду: грохот пузырей
put(under, env(filt(add(resample(splash, .4), resample(storm, .7)[:int(1.5 * SR)], .7), hi=900), .001, .6), 21.6, 1.0, wet=.6)
# пузыри: те же времена, что и в кадре
for k in range(70):
    late = k > 55; tb = 24 + h32(k, 2) * 14 if late else 21.6 + h32(k, 2) * 2.6
    put(under, env(filt(resample(bubble, .5 + 1.1 * h32(k, 3)), hi=2500), .001, .03), tb + .1, .25 + .35 * h32(k, 4), pan=(h32(k, 1) - .5) * 1.4, wet=.6)
# сердце, глухое
hz = add(filt(resample(heart, .8), hi=350), filt(resample(heart, .55), hi=150), .9)
t = 22.4
while t < 39.4:
    put(under, env(hz, .001, .1), t, .55 + .5 * float(sm(29, 38, t)), wet=.15)
    t += 60 / (55 + 60 * float(sm(26, 38, t)))
fin = np.zeros((N, 2))
for t in (40.4, 41.7): s_ = int(t * SR); x_ = env(hz, .001, .1)[:N - s_]; fin[s_:s_ + len(x_)] += x_ * .5
# глаз открылся: протяжный стон кита
moan = filt(add(resample(stretch(choirH, 1.5), .3), resample(stretch(scream, 1.6), .28), .5), hi=600)
put(under, env(moan, .8, 1.5), 27.4, .45, wet=.9)
put(under, env(moan[::-1], .8, 1.5), 31.5, .9, wet=.8)
# светящиеся точки: тонкий хор сверху
put(under, env(filt(resample(choirA[::-1], 1.4), lo=900, hi=5000), 1.5, 2), 30.6, .18, wet=.9)
# пасть открывается: огромный вдох, засасывает
inhale = filt(resample(stretch(breath, 3.5), .28), hi=900)
put(under, env(inhale, 2, .4)[:int(6 * SR)], 33.4, 1.1, wet=.5)
put(under, env(filt(resample(stretch(storm, 1.2), .4), hi=250), 2, .3)[:int(6.2 * SR)], 33.2, 1.0, wet=.3)
for t in (36.1, 37.6, 38.7): put(under, env(filt(resample(bone, .4), hi=1500), .001, .2), t, .6, pan=rng.uniform(-.5, .5), wet=.5)

# ---------- ревербератор и сведение ----------
n_ir = int(3.2 * SR); ir = rng.standard_normal((n_ir, 2)) * np.exp(-np.arange(n_ir) / (.6 * SR))[:, None]; ir = filt(ir, hi=2500)
L = 1 << int(np.ceil(np.log2(N + n_ir)))
rv = np.stack([np.fft.irfft(np.fft.rfft(verb[:, c], L) * np.fft.rfft(ir[:, c], L), L)[:N] for c in (0, 1)], 1)
under = filt(under, hi=1400)
mix = above + under
out = mix + rv / (np.abs(rv).max() + 1e-9) * np.abs(mix).max() * .4
out[ts >= 39.4] = 0
pk = np.abs(out).max(); out = out / pk * 1.5 + filt(fin, hi=300) / (np.abs(fin).max() + 1e-9) * .5
out = np.tanh(out) * .89
subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 's16le', '-ar', str(SR), '-ac', '2', '-i', '-', OUT],
               input=(out * 32767).astype('<i2').tobytes(), check=True)
