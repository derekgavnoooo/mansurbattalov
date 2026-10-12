"""Сборка ролика.  python3 make.py <клип_с_треком.mp4> [out.mp4]

Звук берётся из исходного клипа (отрезок 0:43–1:13), картинка рисуется заново."""
import os
import subprocess
import sys
import tempfile
from multiprocessing import Pool

import lyrics
import scenes

TMP = os.environ.get("FRAMES_DIR") or tempfile.mkdtemp(prefix="daikirai_")


def work(d):
    p = os.path.join(TMP, f"d{d:05d}.png")
    if not os.path.exists(p):
        scenes.render(d * scenes.TWOS).save(p)
    return d


if __name__ == "__main__":
    src = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "daikirai_demo.mp4"
    os.makedirs(TMP, exist_ok=True)
    n_frames = int(lyrics.CLIP_LEN * scenes.FPS)
    n_draw = (n_frames + scenes.TWOS - 1) // scenes.TWOS
    with Pool(os.cpu_count()) as pool:
        for k, _ in enumerate(pool.imap_unordered(work, range(n_draw))):
            if k % 30 == 0:
                print(f"{k}/{n_draw}", flush=True)
    with open(os.path.join(TMP, "list.txt"), "w") as f:
        for d in range(n_draw):
            f.write(f"file 'd{d:05d}.png'\nduration {scenes.TWOS / scenes.FPS:.6f}\n")
        f.write(f"file 'd{n_draw - 1:05d}.png'\n")
    L = lyrics.CLIP_LEN
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error",
                    "-f", "concat", "-safe", "0", "-i", os.path.join(TMP, "list.txt"),
                    "-ss", str(lyrics.CLIP_T0), "-t", str(L), "-i", src,
                    "-map", "0:v", "-map", "1:a",
                    "-vf", f"fps={scenes.FPS},format=yuv420p",
                    "-af", f"afade=t=in:d=0.25,afade=t=out:st={L - 1.2}:d=1.2",
                    "-c:v", "libx264", "-crf", "18", "-preset", "slow", "-tune", "animation",
                    "-c:a", "aac", "-b:a", "192k", "-t", str(L), "-movflags", "+faststart", out], check=True)
    print("done:", out)
