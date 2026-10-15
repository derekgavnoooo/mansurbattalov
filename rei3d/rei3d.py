"""Рэй как 3D-модель в Blender (bpy) с аниме-шейдингом: двухтоновая тень + контур Freestyle.

python3 rei3d.py <папка_для_рендеров>
Модель собирается кодом из примитивов, прядей-трубок и «наклеек» лица; рендер Cycles на CPU.
"""
import math
import os
import sys
import bpy
import numpy as np
from mathutils import Vector, Euler

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
bpy.ops.wm.read_factory_settings(use_empty=True)
SC = bpy.context.scene
COL = SC.collection
HEAD_C = Vector((0, 0, 1.40))
HR = 0.11  # радиус черепа


def hexc(h, a=1.0):
    h = h.lstrip("#")
    c = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    return [x ** 2.2 for x in c] + [a]          # sRGB → linear


# ---------------- материалы ----------------

def toon(name, base, shade, size=0.42):
    """Аниме-тень: тень = shade, на свету = base (резкая граница)."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = hexc(shade)
    tb = nt.nodes.new("ShaderNodeBsdfToon")
    b, s = hexc(base), hexc(shade)
    tb.inputs["Color"].default_value = [max(0.0, b[i] - s[i]) for i in range(3)] + [1]
    tb.inputs["Size"].default_value = size
    tb.inputs["Smooth"].default_value = 0.0
    add = nt.nodes.new("ShaderNodeAddShader")
    nt.links.new(em.outputs[0], add.inputs[0])
    nt.links.new(tb.outputs[0], add.inputs[1])
    nt.links.new(add.outputs[0], out.inputs[0])
    return m


def flat(name, color):
    """Неосвещаемый цвет — для нарисованных глаз, рта, румянца."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = hexc(color)
    nt.links.new(em.outputs[0], out.inputs[0])
    return m


M = dict(
    skin=toon("skin", "#fde9e4", "#efb6b4"), hair=toon("hair", "#f7f3fd", "#b9aee0"),
    tip=toon("tip", "#ff6f8e", "#c8385f"), blazer=toon("blazer", "#5b5f9c", "#353865"),
    shirt=toon("shirt", "#ffffff", "#c9c3df"), tie=toon("tie", "#ef3f62", "#a81f45"),
    skirt=toon("skirt", "#3e3a5c", "#25223a"), sock=toon("sock", "#ffffff", "#c9c3df"),
    shoe=toon("shoe", "#2d2333", "#1b141f"), clip=toon("clip", "#ff3f62", "#b0203f"),
    white=flat("eyewhite", "#ffffff"), lash=flat("lash", "#2b1d27"), irisL=flat("irisL", "#ff3d5c"),
    irisR=flat("irisR", "#3d6bff"), irisD=flat("irisD", "#4a0f28"), irisDR=flat("irisDR", "#14215e"),
    mouth=flat("mouth", "#a83a55"), blush=flat("blush", "#f59aaa"), cross=flat("cross", "#ffd44a"),
)


def obj(name, verts, faces, mat=None, smooth=True, subsurf=0, parent=None, mats=None, face_mat=None):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in verts], [], faces)
    me.update()
    o = bpy.data.objects.new(name, me)
    COL.objects.link(o)
    for mm in (mats or [mat]):
        if mm:
            me.materials.append(mm)
    if face_mat is not None:
        for p, idx in zip(me.polygons, face_mat):
            p.material_index = idx
    if smooth:
        for p in me.polygons:
            p.use_smooth = True
    if subsurf:
        md = o.modifiers.new("sub", "SUBSURF")
        md.levels = subsurf
        md.render_levels = subsurf
    if parent:
        o.parent = parent
    return o


def loft(rings, n=28, cap_top=True, cap_bot=True):
    """Поверхность через кольца [(z, rx, ry, cx, cy, fn)], fn(θ)->множитель радиуса."""
    V, F = [], []
    for z, rx, ry, cx, cy, fn in rings:
        for i in range(n):
            t = 2 * math.pi * i / n
            k = fn(t) if fn else 1.0
            V.append((cx + rx * k * math.cos(t), cy + ry * k * math.sin(t), z))
    for j in range(len(rings) - 1):
        for i in range(n):
            a, b = j * n + i, j * n + (i + 1) % n
            F.append((a, b, b + n, a + n))
    if cap_top:
        F.append(tuple(range(n))[::-1])
    if cap_bot:
        F.append(tuple(range((len(rings) - 1) * n, len(rings) * n)))
    return V, F


def tube(path, radii, n=12, flatten=1.0, up=None):
    """Трубка вдоль ломаной; flatten<1 сплющивает сечение (для прядей)."""
    P = [Vector(p) for p in path]
    V, F = [], []
    for k, p in enumerate(P):
        t = (P[min(k + 1, len(P) - 1)] - P[max(k - 1, 0)]).normalized()
        a = (up or Vector((0, 0, 1))).cross(t)
        if a.length < 1e-4:
            a = Vector((1, 0, 0)).cross(t)
        a.normalize()
        b = t.cross(a).normalized()
        r = radii[k]
        for i in range(n):
            th = 2 * math.pi * i / n
            V.append(p + a * (r * math.cos(th)) + b * (r * flatten * math.sin(th)))
    for j in range(len(P) - 1):
        for i in range(n):
            a_, b_ = j * n + i, j * n + (i + 1) % n
            F.append((a_, b_, b_ + n, a_ + n))
    F.append(tuple(range(n))[::-1])
    F.append(tuple(range((len(P) - 1) * n, len(P) * n)))
    return V, F


# ---------------- голова ----------------

HEAD = bpy.data.objects.new("head_ctrl", None)
COL.objects.link(HEAD)
HEAD.location = HEAD_C


def head_mesh():
    V, F = [], []
    nu, nv = 36, 24
    for j in range(nv + 1):
        ph = math.pi * j / nv
        for i in range(nu):
            th = 2 * math.pi * i / nu
            x, y, z = math.sin(ph) * math.cos(th), math.sin(ph) * math.sin(th), math.cos(ph)
            x, y, z = x * HR * 0.93, y * HR, z * HR * 1.02
            if z < 0:                                    # аниме-подбородок: сужение и вытягивание вниз-вперёд
                t = min(1.0, -z / HR)
                k = 1 - 0.62 * t ** 1.5
                x *= k
                y = y * (1 - 0.35 * t) - 0.012 * t
                z *= 1.06
            V.append((x, y, z))
    for j in range(nv - 3):                      # низ черепа (внутри шеи) не строим — без лишних линий
        for i in range(nu):
            a, b = j * nu + i, j * nu + (i + 1) % nu
            F.append((a, b, b + nu, a + nu))
    return V, F


def face_y(x, z):
    """Глубина передней поверхности лица в точке (x, z) локально — для наклеек."""
    zz = z / 1.06 if z < 0 else z
    t = min(1.0, -zz / HR) if zz < 0 else 0
    k = 1 - 0.62 * t ** 1.5 if zz < 0 else 1
    xs = x / (0.93 * k)
    val = max(1e-6, HR ** 2 - xs ** 2 - (zz / 1.02) ** 2)
    y = -math.sqrt(val)
    if zz < 0:
        y = y * (1 - 0.35 * t) - 0.012 * t
    return y


def decal(name, pts2d, mat, cx, cz, lift, parent, scale=1.45):
    """Плоская «наклейка» на лицо: pts2d в мм относительно центра (cx, cz)."""
    V = []
    for px, pz in pts2d:
        x = cx + px * 0.001 * scale
        z = cz + pz * 0.001 * scale
        V.append((x, face_y(x, z) - lift, z))
    return obj(name, V, [tuple(range(len(V)))], mat, smooth=False, parent=parent)


def ell(rx, rz, n=24, ox=0, oz=0):
    return [(ox + rx * math.cos(2 * math.pi * i / n), oz + rz * math.sin(2 * math.pi * i / n)) for i in range(n)]


def build_head(eyes="open", mouth="smile"):
    v, f = head_mesh()
    obj("head", v, f, M["skin"], subsurf=1, parent=HEAD)
    for s, ir, dk in ((-1, M["irisL"], M["irisD"]), (1, M["irisR"], M["irisDR"])):
        cx, cz = s * 0.040, -0.026
        m = 1 if s > 0 else -1
        X = lambda pts: [(m * x, z) for x, z in pts]
        if eyes == "closed":
            decal(f"lid{s}", X([(-15, 2), (-8, -4), (0, -6), (8, -4), (15, 1), (8, -2), (0, -3), (-8, -1)]), M["lash"], cx, cz, 0.0010, HEAD)
            continue
        white = [(-15, 4), (-12, 9), (-4, 12), (6, 11), (13, 7), (15, 2), (12, -8), (4, -13), (-6, -12), (-13, -6)]
        decal(f"white{s}", X(white), M["white"], cx, cz, 0.0006, HEAD)
        decal(f"iris{s}", X(ell(8.5, 11.5, ox=1, oz=-1)), ir, cx, cz, 0.0009, HEAD)
        decal(f"irisd{s}", X([(-8, 5), (-5, 9), (2, 10.5), (8, 7), (9, 2), (0, 4), (-7, 2)]), dk, cx, cz, 0.0011, HEAD)
        decal(f"pupil{s}", X(ell(3.6, 5.2, ox=1, oz=-1)), dk, cx, cz, 0.0012, HEAD)
        crossp = [(-1, 3), (1, 3), (1, 1), (3, 1), (3, -1), (1, -1), (1, -3), (-1, -3), (-1, -1), (-3, -1), (-3, 1), (-1, 1)]
        decal(f"cross{s}", X([(x + 1, z - 1) for x, z in crossp]), M["cross"], cx, cz, 0.0013, HEAD)
        decal(f"hl{s}", X(ell(2.6, 3.2, ox=-3.5, oz=5)), M["white"], cx, cz, 0.0014, HEAD)
        lash = [(-17, 2), (-14, 8), (-6, 13), (4, 13.5), (12, 10), (16, 4), (13, 7), (4, 10.5), (-6, 10), (-13, 5), (-15, 1)]
        decal(f"lash{s}", X(lash), M["lash"], cx, cz, 0.0016, HEAD)
        decal(f"flick{s}", X([(-15, 3), (-20, 6), (-19, 3), (-16, 1)]), M["lash"], cx, cz, 0.0016, HEAD)
        decal(f"low{s}", X([(-8, -12.5), (0, -14), (7, -12), (0, -12.8)]), M["lash"], cx, cz, 0.0016, HEAD)
        decal(f"brow{s}", X([(-14, 24), (-2, 28), (10, 27), (-2, 26)]), M["lash"], cx, cz, 0.0010, HEAD)
        bl = decal(f"blush{s}", X(ell(8, 3.2, ox=1, oz=-19)), M["blush"], cx, cz, 0.0006, HEAD)
        for k in range(0):
            decal(f"hatch{s}{k}", X([(-3 + 3 * k, -21), (-2.4 + 3 * k, -21), (-0.5 + 3 * k, -17.5), (-1.1 + 3 * k, -17.5)]), M["mouth"], cx, cz, 0.0009, HEAD)
    if mouth == "smile":
        decal("mouth", [(-7, 1), (-3, -1.5), (3, -1.5), (7, 1), (3, -0.3), (-3, -0.3)], M["lash"], 0.0, -0.080, 0.0012, HEAD, scale=0.9)
        decal("fang", [(2, -1.2), (4.5, -1.2), (3.3, -4)], M["white"], 0.0, -0.080, 0.0014, HEAD, scale=0.9)
    else:
        decal("mouth", ell(5, 5.5), M["mouth"], 0.0, -0.090, 0.0012, HEAD)
        decal("mouthl", [(-5, 4), (5, 4), (4, 5.5), (-4, 5.5)], M["lash"], 0.0, -0.090, 0.0014, HEAD)
    decal("nose", [(-0.6, 0), (0.8, 0), (0.5, -2.5), (-0.4, -2.2)], M["mouth"], 0.0, -0.056, 0.0010, HEAD, scale=1.0)


def hair_strand(name, az, p0, p1, w, ext=0.0, curl=0.0, r0=None, lift=0.006, tip_frac=0.0, n=9):
    """Прядь по поверхности головы: азимут az (−90° — перёд), полярный угол от p0 до p1 (градусы)."""
    pts, rad = [], []
    for k in range(n):
        t = k / (n - 1)
        ph = math.radians(p0 + (p1 - p0) * t)
        a = math.radians(az + curl * t * t)
        r = (r0 or HR) + lift + ext * t ** 2
        x, y, z = r * math.sin(ph) * math.cos(a) * 0.95, r * math.sin(ph) * math.sin(a), r * math.cos(ph)
        pts.append((x, y, z))
        rad.append(w * (1 - t ** 1.7) + 0.0006)
    V, F = tube(pts, rad, n=8, flatten=0.32)
    nfaces = len(F)
    seg = 8
    fm = []
    for i in range(nfaces):
        if i < (n - 1) * seg:
            fm.append(1 if (i // seg) >= (n - 1) * (1 - tip_frac) else 0)
        else:
            fm.append(1 if i == nfaces - 1 and tip_frac > 0 else 0)
    return obj(name, V, F, mats=[M["hair"], M["tip"]], face_mat=fm, parent=HEAD, subsurf=1)


def build_hair():
    # объём каре: шапка волос
    V, F = [], []
    nu, nv = 36, 16
    for j in range(nv + 1):
        ph = math.radians(4 + 116 * j / nv)
        for i in range(nu):
            th = 2 * math.pi * i / nu
            front = max(0.0, -math.sin(th))            # спереди шапка короче — открывает лицо
            ph_eff = ph
            r = HR * 1.07
            x, y, z = r * math.sin(ph_eff) * math.cos(th) * 0.97, r * math.sin(ph_eff) * math.sin(th), r * math.cos(ph_eff)
            V.append((x, y + 0.004, z))
    keep = []
    for j in range(nv):
        for i in range(nu):
            th = 2 * math.pi * (i + 0.5) / nu
            ph = 4 + 116 * (j + 0.5) / nv
            front = -math.sin(th)
            if front > 0.45 and ph > 62:               # вырез под лицо
                continue
            a, b = j * nu + i, j * nu + (i + 1) % nu
            F.append((a, b, b + nu, a + nu))
    obj("haircap", V, F, M["hair"], parent=HEAD, subsurf=1)
    # чёлка
    for k, (az, p1, w, curl) in enumerate([(-130, 80, 0.014, 6), (-118, 84, 0.015, 4), (-106, 88, 0.016, 3),
                                           (-96, 92, 0.014, -2), (-90, 100, 0.012, 0), (-84, 92, 0.014, 2),
                                           (-74, 88, 0.016, -3), (-62, 84, 0.015, -4), (-50, 80, 0.014, -6)]):
        hair_strand(f"bang{k}", az, 8, p1, w, ext=0.008, curl=curl, lift=0.010)
    # боковые и задние пряди каре с красными кончиками и разлётом
    for k, az in enumerate(list(range(-160, 30, 13)) + list(range(30, 200, 15))):
        if -55 > az > -125:
            continue
        flare = 0.022 if abs(math.sin(math.radians(az))) < 0.7 else 0.012
        hair_strand(f"bob{k}", az, 10, 128, 0.022, ext=flare, curl=(7 if az % 2 else -7), lift=0.012, tip_frac=0.3)
    # пряди, обрамляющие лицо
    for s in (-1, 1):
        hair_strand(f"lock{s}", -90 + s * 52, 30, 140, 0.016, ext=0.004, curl=-s * 4, lift=0.006, tip_frac=0.3)
    # ахоге
    pts = [(0.0, -0.02, HR + 0.006), (0.004, -0.01, HR + 0.03), (0.014, 0.0, HR + 0.05), (0.03, 0.004, HR + 0.055),
           (0.04, 0.0, HR + 0.045)]
    v, f = tube(pts, [0.006, 0.005, 0.004, 0.0025, 0.0006], n=8, flatten=0.4)
    obj("ahoge", v, f, M["hair"], parent=HEAD, subsurf=1)
    # заколка-крестик
    for k, a in enumerate((0.7, -0.7)):
        c = Vector((-0.052, -0.088, 0.055))
        d = Vector((math.cos(a), 0, math.sin(a))) * 0.012
        v, f = tube([c - d, c + d], [0.0035, 0.0035], n=8)
        obj(f"clip{k}", v, f, M["clip"], parent=HEAD)


# ---------------- тело ----------------

def build_body(arm_pose="down"):
    v, f = loft([(1.24, 0.026, 0.026, 0, 0, None), (1.35, 0.026, 0.026, 0, 0, None)], n=16, cap_top=False, cap_bot=False)
    obj("neck", v, f, M["skin"], subsurf=1)
    lap = lambda t: 1.0 + 0.04 * math.cos(2 * t)
    rings = [(1.255, 0.07, 0.05, 0, 0, None), (1.238, 0.15, 0.075, 0, 0, lap), (1.18, 0.158, 0.088, 0, 0, None),
             (1.10, 0.148, 0.086, 0, 0, None), (1.02, 0.132, 0.078, 0, 0, None), (0.94, 0.142, 0.084, 0, 0, None)]
    v, f = loft(rings, n=32)
    obj("blazer", v, f, M["blazer"], subsurf=2)
    # рубашка и галстук в вырезе — тонкие пластины перед грудью
    shirt = [(-0.032, -0.072, 1.255), (0.032, -0.072, 1.255), (0.0, -0.093, 1.12)]
    obj("shirt", shirt, [(0, 1, 2)], M["shirt"], smooth=False)
    tie = [(-0.008, -0.078, 1.24), (0.008, -0.078, 1.24), (0.011, -0.088, 1.15), (0.0, -0.090, 1.135), (-0.011, -0.088, 1.15)]
    obj("tie", tie, [(0, 1, 2, 3, 4)], M["tie"], smooth=False)
    for s in (-1, 1):
        obj(f"lapel{s}", [(s * 0.032, -0.072, 1.256), (s * 0.075, -0.07, 1.245), (s * 0.02, -0.092, 1.10), (s * 0.004, -0.093, 1.12)],
            [(0, 1, 2, 3)], M["blazer"], smooth=False)
        obj(f"collar{s}", [(s * 0.004, -0.074, 1.262), (s * 0.034, -0.07, 1.262), (s * 0.02, -0.082, 1.225)], [(0, 1, 2)], M["shirt"], smooth=False)
    # руки: длинные рукава, кисти выглядывают
    for s in (-1, 1):
        if arm_pose == "down":
            path = [(s * 0.125, 0.0, 1.20), (s * 0.165, 0.0, 1.12), (s * 0.185, -0.02, 0.96), (s * 0.18, -0.04, 0.86)]
        else:
            path = [(s * 0.15, 0.0, 1.235), (s * 0.20, -0.03, 1.12), (s * 0.12, -0.10, 1.13), (s * 0.05, -0.13, 1.20)]
        v, f = tube(path, [0.042, 0.040, 0.037, 0.040], n=14)
        f = f[:-2] + [f[-1]]                     # без торца у плеча — рукав «растёт» из пиджака
        obj(f"sleeve{s}", v, f, M["blazer"], subsurf=1)
        end = Vector(path[-1])
        d = (end - Vector(path[-2])).normalized()
        v, f = loft([(0, 0.022, 0.016, 0, 0, None), (0.03, 0.02, 0.014, 0, 0, None), (0.045, 0.01, 0.008, 0, 0, None)], n=12)
        h = obj(f"hand{s}", v, f, M["skin"], subsurf=1)
        h.location = end + d * 0.005
        h.rotation_euler = Vector((0, 0, 1)).rotation_difference(d).to_euler()
    # юбка в складку
    pleat = lambda t: 1.0 + 0.05 * abs(math.sin(8 * t))
    v, f = loft([(0.95, 0.142, 0.085, 0, 0, None), (0.86, 0.19, 0.13, 0, 0, pleat), (0.77, 0.23, 0.17, 0, 0, pleat)],
                n=48, cap_top=False, cap_bot=False)
    obj("skirt", v, f, M["skirt"], subsurf=1)
    for s in (-1, 1):
        v, f = tube([(s * 0.06, 0.0, 0.86), (s * 0.065, 0.0, 0.50), (s * 0.07, 0.01, 0.12)], [0.05, 0.038, 0.03], n=14)
        obj(f"leg{s}", v, f, M["skin"], subsurf=1)
        v, f = tube([(s * 0.065, 0.0, 0.46), (s * 0.07, 0.01, 0.12)], [0.040, 0.033], n=14)
        obj(f"sock{s}", v, f, M["sock"], subsurf=1)
        v, f = tube([(s * 0.07, 0.02, 0.06), (s * 0.072, -0.08, 0.05)], [0.04, 0.035], n=14, flatten=0.7)
        obj(f"shoe{s}", v, f, M["shoe"], subsurf=1)


# ---------------- сцена ----------------

def setup_render(w=1280, h=720, samples=24):
    r = SC.render
    r.engine = "CYCLES"
    SC.cycles.device = "CPU"
    SC.cycles.samples = samples
    SC.cycles.use_denoising = True
    r.resolution_x, r.resolution_y = w, h
    r.film_transparent = True
    SC.view_settings.view_transform = "Standard"
    r.use_freestyle = True
    bpy.context.view_layer.use_freestyle = True
    r.line_thickness_mode = "ABSOLUTE"
    r.line_thickness = 2.2
    fs = bpy.context.view_layer.freestyle_settings
    fs.crease_angle = math.radians(135)
    ls = fs.linesets.new("outline")
    ls.select_by_visibility = True
    ls.select_by_edge_types = True
    ls.select_silhouette = True
    ls.select_border = True
    ls.select_crease = True
    ls.select_external_contour = True
    ls.linestyle.color = hexc("#2b1d27")[:3]
    ls.linestyle.thickness = 2.2
    world = bpy.data.worlds.new("w")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[1].default_value = 0.0
    SC.world = world
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
    COL.objects.link(sun)
    sun.data.energy = math.pi
    sun.data.angle = 0.02
    sun.rotation_euler = Euler((math.radians(50), math.radians(-25), math.radians(-30)))


def camera(loc, target, lens=50):
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    COL.objects.link(cam)
    cam.data.lens = lens
    cam.location = loc
    d = Vector(target) - Vector(loc)
    cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    SC.camera = cam
    return cam


def shot(name, loc, target, lens, head_rot=(0, 0, 0), roll=0.0):
    HEAD.rotation_euler = Euler([math.radians(a) for a in head_rot])
    cam = camera(loc, target, lens)
    cam.rotation_euler.rotate_axis("Z", math.radians(roll))
    SC.render.filepath = os.path.join(OUT, name + ".png")
    bpy.ops.render.render(write_still=True)
    bpy.data.objects.remove(cam)


if __name__ == "__main__":
    build_head()
    build_hair()
    build_body()
    setup_render()
    shot("A_bust_34", (-0.55, -1.15, 1.42), (0, 0, 1.33), 70, head_rot=(0, 0, -12))
    shot("B_low_angle", (0.28, -0.80, 1.02), (0, 0, 1.36), 30, head_rot=(10, 0, 10), roll=-10)
    shot("C_high_angle", (0.22, -0.80, 2.05), (0, 0, 1.30), 42, head_rot=(-26, 0, -8), roll=12)
