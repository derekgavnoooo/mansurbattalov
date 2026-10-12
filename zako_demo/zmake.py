"""Сборка: python3 zmake.py <клип «雑魚».mp4> out.mp4"""
import os
import subprocess
import sys
import tempfile
from multiprocessing import Pool

import zscenes as Z

TMP = os.environ.get("FRAMES_DIR") or tempfile.mkdtemp(prefix="zako_")


def work(d):
    p = os.path.join(TMP, f"z{d:05d}.png")
    if not os.path.exists(p):
        Z.render(d * Z.TWOS).save(p)
    return d


if __name__ == "__main__":
    src, out = sys.argv[1], sys.argv[2]
    os.makedirs(TMP, exist_ok=True)
    n_draw = int(Z.DUR * Z.FPS) // Z.TWOS
    with Pool(os.cpu_count()) as pool:
        for k, _ in enumerate(pool.imap_unordered(work, range(n_draw))):
            if k % 60 == 0:
                print(f"{k}/{n_draw}", flush=True)
    with open(os.path.join(TMP, "list.txt"), "w") as f:
        for d in range(n_draw):
            f.write(f"file 'z{d:05d}.png'\nduration {Z.TWOS / Z.FPS:.6f}\n")
        f.write(f"file 'z{n_draw - 1:05d}.png'\n")
    D = Z.DUR
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", os.path.join(TMP, "list.txt"),
                    "-ss", str(Z.T0), "-t", str(D), "-i", src, "-map", "0:v", "-map", "1:a",
                    "-vf", f"fps={Z.FPS},format=yuv420p", "-af", f"afade=t=in:d=0.3,afade=t=out:st={D - 1.0}:d=1.0",
                    "-c:v", "libx264", "-crf", "20", "-preset", "slow", "-tune", "animation", "-c:a", "aac", "-b:a", "192k",
                    "-t", str(D), "-movflags", "+faststart", out], check=True)
    print("done", out)
