"""Собирает ролик: звук + кадры + ffmpeg.  python3 make.py [out.mp4]"""
import os
import subprocess
import sys
import tempfile
from multiprocessing import Pool

import render
import song


def work(d):
    path = os.path.join(TMP, f"d{d:05d}.png")
    if not os.path.exists(path):
        render.render(d * render.TWOS).save(path)
    return d


TMP = os.environ.get("FRAMES_DIR") or tempfile.mkdtemp(prefix="nota_")

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "nota_demo.mp4"
    os.makedirs(TMP, exist_ok=True)
    n_frames = int(song.TOTAL * render.FPS)
    n_draw = (n_frames + render.TWOS - 1) // render.TWOS
    wav = os.path.join(TMP, "audio.wav")
    song.render_audio(wav, render.schedule()["audio"])
    print("audio ok")
    with Pool(os.cpu_count()) as pool:
        for k, _ in enumerate(pool.imap_unordered(work, range(n_draw))):
            if k % 20 == 0:
                print(f"{k}/{n_draw}", flush=True)
    # каждый рисунок держится 2 кадра (анимация «по двойкам»)
    with open(os.path.join(TMP, "list.txt"), "w") as f:
        for d in range(n_draw):
            f.write(f"file 'd{d:05d}.png'\nduration {render.TWOS / render.FPS:.6f}\n")
        f.write(f"file 'd{n_draw - 1:05d}.png'\n")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                    "-i", os.path.join(TMP, "list.txt"), "-i", wav,
                    "-vf", f"fps={render.FPS},format=yuv420p", "-c:v", "libx264", "-crf", "19",
                    "-preset", "slow", "-tune", "animation", "-c:a", "aac", "-b:a", "192k",
                    "-shortest", "-movflags", "+faststart", out], check=True)
    print("done:", out)
