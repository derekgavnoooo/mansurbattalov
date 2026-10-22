"""Общий таймлайн (без bpy): доли, ключевые моменты, огибающая вокала."""
import json
import math
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AN = json.load(open(os.path.join(HERE, "..", "daikirai_demo", "analysis.json")))
B = np.array(AN["beats"])
ENV = np.array(AN["env"])
FPS = 24
T0, T1 = 0.4, 18.6
E8 = (B[1] - B[0]) / 2
DAI = [B[6] + E8 * k for k in range(1, 6)]
KIRAI = B[9]
BREAK0, DROP = 15.30, 16.90
FLASHES = [(B[3], 0.85, 6), (KIRAI, 0.6, 8), (DROP, 0.95, 4), (B[10] + 0.93, 0.35, 9), (9.28, 0.35, 9), (12.97, 0.45, 8)] + \
          [(d, 0.15, 14) for d in DAI]


def pulse(t, dec=6.0):
    if BREAK0 <= t < DROP or t < B[3]:
        return 0.0
    i = int(np.searchsorted(B, t, side="right")) - 1
    return math.exp(-(t - B[i]) * dec) if i >= 0 else 0.0


def env_at(t):
    i = int(t / AN["env_dt"])
    return float(np.clip(ENV[i], 0, 1.2)) if 0 <= i < len(ENV) else 0.0
