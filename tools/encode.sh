#!/bin/sh
# Сборка видео: encode.sh <папка_кадров> <звук.wav> <выход.mp4>
# Кадры 180×320 при 12 fps -> 1080×1920, 30 fps, увеличение без сглаживания.
ffmpeg -v error -y -framerate 12 -i "$1/%05d.png" -i "$2" -vf "scale=1080:1920:flags=neighbor,fps=30" \
  -c:v libx264 -crf 28 -preset slow -tune grain -x264-params keyint=60 -pix_fmt yuv420p \
  -c:a aac -b:a 192k -shortest -movflags +faststart "$3"
