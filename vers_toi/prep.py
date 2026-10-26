"""Карты глубины для всех фонов: python prep.py <assets>"""
import sys,glob,os
from transformers import pipeline
from PIL import Image
D=sys.argv[1];de=pipeline('depth-estimation',model='depth-anything/Depth-Anything-V2-Small-hf')
for f in sorted(glob.glob(f'{D}/bg_*.png')):
    o=f.replace('/bg_','/dep_')
    if not os.path.exists(o):de(Image.open(f))['depth'].save(o);print('depth',os.path.basename(f),flush=True)
