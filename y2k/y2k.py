"""Y2K-демо в серебре: жидкий хром, хромовые звёзды и текст, голографические пузыри,
сетка-пол, UI нулевых поверх. Рендер Blender Cycles (CPU), монтаж по долям Daidaidai.

python3 y2k.py frames_dir [start_frame end_frame]
"""
import json
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Euler, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else "."
FPS = 24
AN = json.load(open(os.path.join(HERE, "..", "daikirai_demo", "analysis.json")))
B = np.array(AN["beats"])
T0, T1 = 0.4, 18.6                      # окно (сек от 0:43.0 трека)
E8 = (B[1] - B[0]) / 2
DAI = [B[6] + E8 * k for k in range(1, 6)]
KIRAI = B[9]
BREAK0, DROP = 15.30, 16.90
FONT_JP = os.path.join(HERE, "..", "daikirai_demo", "fonts", "dela-gothic-one-japanese-400-normal.ttf")
FONT_LAT = "/usr/share/fonts/opentype/inter/InterDisplay-Bold.otf"

bpy.ops.wm.read_factory_settings(use_empty=True)
SC = bpy.context.scene
COL = SC.collection


def lin(h):
    h = h.lstrip("#")
    return [(int(h[i:i + 2], 16) / 255) ** 2.2 for i in (0, 2, 4)] + [1.0]


# ---------------- материалы ----------------

def principled(name, color, metallic=1.0, rough=0.05, thin=0.0, coat=0.0, emission=None, estr=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    p = m.node_tree.nodes["Principled BSDF"]
    p.inputs["Base Color"].default_value = lin(color)
    p.inputs["Metallic"].default_value = metallic
    p.inputs["Roughness"].default_value = rough
    if thin:
        p.inputs["Thin Film Thickness"].default_value = thin
        p.inputs["Thin Film IOR"].default_value = 1.45
    if coat:
        p.inputs["Coat Weight"].default_value = coat
    if emission:
        p.inputs["Emission Color"].default_value = lin(emission)
        p.inputs["Emission Strength"].default_value = estr
    return m


CHROME = principled("chrome", "#e6e9f0", 1.0, 0.035)
CHROME_SOFT = principled("chrome_soft", "#d8dbe6", 1.0, 0.16)
HOLO = principled("holo", "#f2f2ff", 1.0, 0.06, thin=520)
PEARL = principled("pearl", "#ffffff", 0.35, 0.12, thin=380, coat=1.0)


def floor_mat():
    m = bpy.data.materials.new("floor")
    m.use_nodes = True
    nt = m.node_tree
    p = nt.nodes["Principled BSDF"]
    p.inputs["Base Color"].default_value = lin("#5a5e6b")
    p.inputs["Metallic"].default_value = 1.0
    p.inputs["Roughness"].default_value = 0.18
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Object"], sep.inputs[0])
    lines = []
    for axis in ("X", "Y"):
        mul = nt.nodes.new("ShaderNodeMath"); mul.operation = "MULTIPLY"; mul.inputs[1].default_value = 0.8
        nt.links.new(sep.outputs[axis], mul.inputs[0])
        fr = nt.nodes.new("ShaderNodeMath"); fr.operation = "PINGPONG"; fr.inputs[1].default_value = 0.5
        nt.links.new(mul.outputs[0], fr.inputs[0])
        lt = nt.nodes.new("ShaderNodeMath"); lt.operation = "LESS_THAN"; lt.inputs[1].default_value = 0.025
        nt.links.new(fr.outputs[0], lt.inputs[0])
        lines.append(lt)
    mx = nt.nodes.new("ShaderNodeMath"); mx.operation = "MAXIMUM"
    nt.links.new(lines[0].outputs[0], mx.inputs[0]); nt.links.new(lines[1].outputs[0], mx.inputs[1])
    em = nt.nodes.new("ShaderNodeMath"); em.operation = "MULTIPLY"; em.inputs[1].default_value = 1.4
    nt.links.new(mx.outputs[0], em.inputs[0])
    p.inputs["Emission Color"].default_value = lin("#ffffff")
    nt.links.new(em.outputs[0], p.inputs["Emission Strength"])
    return m


# ---------------- окружение для отражений ----------------

def world():
    w = bpy.data.worlds.new("w")
    w.use_nodes = True
    nt = w.node_tree
    bg = nt.nodes["Background"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Generated"], sep.inputs[0])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    cr = ramp.color_ramp
    cr.elements[0].position, cr.elements[0].color = 0.0, lin("#1a1c24")
    cr.elements[1].position, cr.elements[1].color = 1.0, lin("#5d6274")
    e = cr.elements.new(0.47); e.color = lin("#3a3d4a")
    e = cr.elements.new(0.51); e.color = lin("#f4f6fb")          # светлая полоса горизонта
    e = cr.elements.new(0.55); e.color = lin("#c3c8d6")
    e = cr.elements.new(0.70); e.color = lin("#e9d9ef")          # лёгкий розовый перелив
    e = cr.elements.new(0.85); e.color = lin("#9aa3bb")
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.inputs[1].default_value, mr.inputs[2].default_value = -1.0, 1.0
    nt.links.new(sep.outputs["Z"], mr.inputs[0])
    nt.links.new(mr.outputs[0], ramp.inputs[0])
    nt.links.new(ramp.outputs[0], bg.inputs[0])
    bg.inputs[1].default_value = 1.0
    SC.world = w


def softbox(name, loc, rot, size, color, strength):
    """Светящиеся панели: в кадре их не видно, но они рисуют блики на хроме."""
    me = bpy.data.meshes.new(name)
    sx, sy = size
    me.from_pydata([(-sx, -sy, 0), (sx, -sy, 0), (sx, sy, 0), (-sx, sy, 0)], [], [(0, 1, 2, 3)])
    o = bpy.data.objects.new(name, me)
    COL.objects.link(o)
    o.location, o.rotation_euler = loc, Euler(rot)
    m = bpy.data.materials.new(name + "_m")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs[0].default_value = lin(color)
    em.inputs[1].default_value = strength
    nt.links.new(em.outputs[0], out.inputs[0])
    me.materials.append(m)
    o.visible_camera = False
    o.visible_shadow = False
    return o


# ---------------- объекты ----------------

def star_mesh(name, r1=1.0, r2=0.22, depth=0.18, points=4):
    V, F = [], []
    n = points * 2
    for k in range(n):
        a = math.pi / 2 + k * math.pi / points
        r = r1 if k % 2 == 0 else r2
        V.append((r * math.cos(a), r * math.sin(a), 0))
    V += [(0, 0, depth), (0, 0, -depth)]
    for k in range(n):
        F.append((k, (k + 1) % n, n))
        F.append(((k + 1) % n, k, n + 1))
    me = bpy.data.meshes.new(name)
    me.from_pydata(V, [], F)
    o = bpy.data.objects.new(name, me)
    COL.objects.link(o)
    me.materials.append(CHROME)
    return o


def text_obj(name, body, font, size=1.0, extrude=0.12, bevel=0.035, mat=CHROME):
    cu = bpy.data.curves.new(name, "FONT")
    cu.body = body
    cu.font = bpy.data.fonts.load(font)
    cu.size = size
    cu.extrude = extrude
    cu.bevel_depth = bevel
    cu.bevel_resolution = 4
    cu.align_x, cu.align_y = "CENTER", "CENTER"
    o = bpy.data.objects.new(name, cu)
    COL.objects.link(o)
    cu.materials.append(mat)
    return o


def sphere(name, r, mat, loc=(0, 0, 0)):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=r, location=loc, segments=48, ring_count=24)
    o = bpy.context.object
    o.name = name
    bpy.ops.object.shade_smooth()
    o.data.materials.append(mat)
    return o


def torus(name, R, r, mat):
    bpy.ops.mesh.primitive_torus_add(major_radius=R, minor_radius=r, major_segments=96, minor_segments=24)
    o = bpy.context.object
    o.name = name
    bpy.ops.object.shade_smooth()
    o.data.materials.append(mat)
    return o


def build():
    world()
    softbox("sb_top", (0, 0, 9), (0, 0, 0), (6, 0.6), "#ffffff", 14)
    softbox("sb_l", (-7, -2, 3), (0, math.radians(-70), 0), (0.5, 4), "#ffffff", 10)
    softbox("sb_r", (7, -1, 3), (0, math.radians(70), 0), (0.5, 4), "#ff9bd9", 7)
    softbox("sb_b", (0, 8, 2), (math.radians(90), 0, 0), (5, 0.15), "#9fe6ff", 9)
    softbox("sb_f", (0, -9, 4), (math.radians(-70), 0, 0), (3, 0.3), "#ffffff", 8)
    me = bpy.data.meshes.new("floor")
    me.from_pydata([(-40, -40, 0), (40, -40, 0), (40, 40, 0), (-40, 40, 0)], [], [(0, 1, 2, 3)])
    fl = bpy.data.objects.new("floor", me)
    COL.objects.link(fl)
    me.materials.append(floor_mat())
    # жидкий хром: метаболы
    mb = bpy.data.metaballs.new("blob")
    mb.resolution = 0.08
    mb.render_resolution = 0.06
    mb.threshold = 0.6
    blob = bpy.data.objects.new("blob", mb)
    COL.objects.link(blob)
    mb.materials.append(CHROME)
    for _ in range(7):
        el = mb.elements.new()
        el.radius = 0.8
    # звёзды-искры, кольца, пузыри, текст
    stars = [star_mesh(f"star{k}") for k in range(6)]
    rings = [torus(f"ring{k}", 2.2 + 0.5 * k, 0.06, CHROME) for k in range(3)]
    bubbles = [sphere(f"bub{k}", 0.25 + 0.12 * (k % 3), HOLO if k % 2 else PEARL) for k in range(10)]
    dai = [text_obj(f"dai{k}", "ダイ", FONT_JP, 0.9) for k in range(5)]
    kirai = text_obj("kirai", "ダイキライ", FONT_JP, 1.25, extrude=0.18, bevel=0.05)
    y2k = text_obj("y2k", "Y2K", FONT_LAT, 2.4, extrude=0.35, bevel=0.06)
    y2k.data.shear = 0.25
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    COL.objects.link(cam)
    SC.camera = cam
    return dict(blob=blob, mb=mb, stars=stars, rings=rings, bubbles=bubbles, dai=dai, kirai=kirai, y2k=y2k, cam=cam)


def look(cam, loc, target, roll=0.0, lens=35):
    cam.location = Vector(loc)
    d = Vector(target) - Vector(loc)
    q = d.to_track_quat("-Z", "Y")
    cam.rotation_euler = q.to_euler()
    cam.rotation_euler.rotate_axis("Z", math.radians(roll))
    cam.data.lens = lens


def ease(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def pulse(t, dec=6.0):
    if BREAK0 <= t < DROP or t < B[3]:
        return 0.0
    i = int(np.searchsorted(B, t, side="right")) - 1
    return math.exp(-(t - B[i]) * dec) if i >= 0 else 0.0


def show(o, v):
    o.hide_render = not v


def face_cam(o, cam, extra_roll=0.0):
    o.rotation_euler = cam.rotation_euler.copy()
    o.rotation_euler.rotate_axis("Z", math.radians(extra_roll))


def pose(S, t):
    """Расставить всё для момента t (сек от 0:43.0)."""
    slow = 0.35 if BREAK0 <= t < DROP else 1.0
    tt = t if t < BREAK0 else (BREAK0 + (t - BREAK0) * 0.35 if t < DROP else BREAK0 + (DROP - BREAK0) * 0.35 + (t - DROP))
    p = pulse(t)
    # капля: элементы кружат и сливаются, на долю — вздох
    on_blob = t >= B[3] - 0.05
    show(S["blob"], on_blob)
    burst = ease((t - DROP) / 0.5) if t >= DROP else 0.0
    for k, el in enumerate(S["mb"].elements):
        a = tt * (0.9 + 0.17 * k) + k * 0.9
        rr = 0.55 + 0.25 * math.sin(tt * 1.3 + k) + burst * (1.6 + 0.4 * k)
        el.co = (rr * math.cos(a), rr * math.sin(a) * 0.8, 1.6 + 0.45 * math.sin(tt * 1.7 + k * 1.3) + burst * 0.3 * math.sin(k))
        el.radius = (0.75 + 0.12 * math.sin(tt * 2 + k)) * (1 + 0.18 * p) * (1 - 0.25 * burst)
    # кольца
    for k, r in enumerate(S["rings"]):
        show(r, t >= B[10] + 0.9)
        r.location = (0, 0, 1.6)
        r.rotation_euler = Euler((math.radians(70 + 12 * k) + tt * 0.25 * (k + 1), tt * 0.4 * (-1) ** k, tt * 0.2))
        r.scale = [1 + 0.05 * p] * 3
    # звёзды-искры
    for k, s in enumerate(S["stars"]):
        show(s, True if t < B[3] else (t >= B[10] + 0.9))
        if t < B[3]:
            s.location = (0, 0, 1.6) if k == 0 else (0, 0, -50)
            s.rotation_euler = Euler((0, tt * 1.4, 0))
            s.scale = [0.9] * 3
        else:
            a = tt * 0.6 + k * math.tau / 6
            s.location = (3.4 * math.cos(a), 3.4 * math.sin(a), 1.8 + 0.8 * math.sin(tt * 1.1 + k))
            s.rotation_euler = Euler((math.radians(90), 0, tt * 2.5 + k))
            s.scale = [0.45 + 0.2 * p] * 3
    # пузыри поднимаются
    for k, b in enumerate(S["bubbles"]):
        show(b, t >= B[15])
        ph = (tt * 0.35 + k * 0.137) % 1.0
        a = k * 2.399
        b.location = (2.6 * math.cos(a) + 0.3 * math.sin(tt + k), 2.6 * math.sin(a) - 0.5, -0.3 + ph * 5.5)
    # «ダイ» × 5 и «ダイキライ»
    spots = [(-2.0, -1.5, 3.0), (2.0, -1.5, 3.1), (-2.0, -1.5, 0.9), (2.0, -1.5, 0.8), (0, -1.8, 3.7)]
    for k, d in enumerate(S["dai"]):
        on = DAI[k] <= t < KIRAI + 0.12
        show(d, on)
        if on:
            age = t - DAI[k]
            sc = 1.0 + 0.6 * math.exp(-age * 14)
            d.location = spots[k]
            d.rotation_euler = Euler((math.radians(90), 0, math.radians((-10, 12, 8, -14, 4)[k])))
            d.scale = [0.6 * sc] * 3
    kon = (KIRAI <= t < B[10] + 0.9) or t >= DROP
    show(S["kirai"], kon)
    if kon:
        age = t - (KIRAI if t < DROP else DROP)
        S["kirai"].location = (0, -2.2, 1.6 if t < DROP else 3.1)
        S["kirai"].rotation_euler = Euler((math.radians(90), 0, math.radians(-3 + 3 * math.sin(tt))))
        S["kirai"].scale = [(1.0 + 0.5 * math.exp(-age * 10)) * (1 if t < DROP else 0.85)] * 3
    y_on = 12.97 <= t < BREAK0
    show(S["y2k"], y_on)
    if y_on:
        S["y2k"].location = (0, -1.0, 2.0)
        S["y2k"].rotation_euler = Euler((math.radians(90), 0, math.radians(-25) + (t - 12.97) * 0.35))
        S["y2k"].scale = [1 + 0.08 * p] * 3
    # камера
    cam = S["cam"]
    shake = Vector(np.random.default_rng(int(t * 100)).normal(0, 1, 3)) * 0.04 * (p + (2.5 * math.exp(-(t - KIRAI) * 6) if t >= KIRAI else 0)
                                                                            + (3 * math.exp(-(t - DROP) * 4) if t >= DROP else 0))
    if t < B[3]:                                           # интро: крупно искра
        k = (t - T0) / (B[3] - T0)
        look(cam, (0.6 - 0.6 * k, -3.2 + 0.8 * k, 1.9), (0, 0, 1.6), roll=8 * (1 - k), lens=50)
    elif t < KIRAI + 0.9:                                  # капля, облёт
        a = -math.pi / 2 + (t - B[3]) * 0.35
        look(cam, Vector((7.0 * math.cos(a), 7.0 * math.sin(a), 2.4)) + shake, (0, 0, 1.8), roll=-4, lens=35 * (1 + 0.04 * p))
    elif t < 9.28:                                         # кольца и звёзды, медленный облёт сверху
        a = -math.pi / 2 + 0.6 + (t - KIRAI) * 0.22
        look(cam, Vector((8.5 * math.cos(a), 8.5 * math.sin(a), 4.2)) + shake, (0, 0, 1.5), roll=6, lens=32)
    elif t < 12.97:                                        # низко над сеткой, пузыри
        a = -math.pi / 2 - 0.4 + (t - 9.28) * 0.3
        look(cam, Vector((6.0 * math.cos(a), 6.0 * math.sin(a), 0.6)) + shake, (0, 0, 2.2), roll=-8, lens=24)
    elif t < BREAK0:                                       # Y2K крупно
        look(cam, Vector((1.2, -7.5 + (t - 12.97) * 0.5, 2.3)) + shake, (0, 0, 2.0), roll=3, lens=40)
    elif t < DROP:                                         # брейк: медленный наезд на хром
        k = (t - BREAK0) / (DROP - BREAK0)
        look(cam, (0.8 - 0.3 * k, -4.5 + 2.0 * ease(k), 2.0), (0, 0, 1.7), roll=-10 * k, lens=45 + 25 * k)
    else:                                                  # дроп: отъезд, взрыв капли
        k = ease((t - DROP) / 1.2)
        look(cam, Vector((0, -5.0 - 5.0 * k, 2.2 + 1.5 * k)) + shake, (0, 0, 2.0), roll=0, lens=30)
    # надписи — в пространстве камеры (всегда в кадре, лицом к зрителю)
    bpy.context.view_layer.update()
    M = cam.matrix_world
    fwd = -(M.to_3x3() @ Vector((0, 0, 1)))
    right = M.to_3x3() @ Vector((1, 0, 0))
    up = M.to_3x3() @ Vector((0, 1, 0))
    spots2d = [(-1.15, 0.55), (1.1, 0.6), (-1.2, -0.45), (1.15, -0.5), (0.0, 0.85)]
    for k, d in enumerate(S["dai"]):
        x, y = spots2d[k]
        d.location = cam.location + fwd * 4.0 + right * x + up * y
        face_cam(d, cam, (-10, 12, 8, -14, 4)[k])
        d.scale = [s_ * 0.55 for s_ in d.scale]
    kk = S["kirai"]
    kk.location = cam.location + fwd * 4.5 + up * (-0.65 if t < DROP else 0.75)
    face_cam(kk, cam, -3 + 3 * math.sin(tt))
    kk.scale = [s_ * 0.62 for s_ in kk.scale]


def setup_render(w=960, h=540, samples=10):
    r = SC.render
    r.engine = "CYCLES"
    SC.cycles.device = "CPU"
    SC.cycles.samples = samples
    SC.cycles.use_denoising = True
    SC.cycles.max_bounces = 4
    SC.cycles.glossy_bounces = 3
    SC.cycles.diffuse_bounces = 1
    SC.cycles.transmission_bounces = 2
    SC.render.use_persistent_data = True
    r.resolution_x, r.resolution_y = w, h
    SC.view_settings.view_transform = "AgX"
    SC.view_settings.look = "AgX - Punchy"
    r.image_settings.file_format = "PNG"


if __name__ == "__main__":
    S = build()
    setup_render()
    if len(sys.argv) > 3:
        f0, f1 = int(sys.argv[2]), int(sys.argv[3])
    else:
        f0, f1 = 0, int((T1 - T0) * FPS)
    times = [float(x) for x in os.environ.get("Y2K_TIMES", "").split(",") if x]
    frames = [(i, T0 + i / FPS) for i in range(f0, f1)] if not times else [(int(round((x - T0) * FPS)), x) for x in times]
    for i, t in frames:
        path = os.path.join(OUT, f"r{i:05d}.png")
        if os.path.exists(path):
            continue
        pose(S, t)
        SC.render.filepath = path
        bpy.ops.render.render(write_still=True)
