"""Нарезка одной картинки на слои: голова (с волосами), левая и правая рука, тело.
Области находятся заливкой между линиями рисунка (как «волшебная палочка»), по плечам ставятся
разрезы. Закрытые места тела дорисовываются: шея под головой, контур бока под руками.
"""
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

# точки-«затравки» внутри областей (координаты исходной картинки 1143×2000)
ARM_SEEDS = {
    "L": [(380, 900), (390, 800), (375, 1000), (370, 1163), (380, 1205), (331, 1052), (331, 1102), (400, 740)],
    "R": [(745, 800), (742, 900), (752, 1000), (765, 1163), (755, 1205), (792, 1058), (796, 1108), (730, 740)],
}
SHOULDER = {"L": (412, 712), "R": (722, 712)}
NECK_PIVOT = (565, 628)


def cuts(barrier):
    d = ImageDraw.Draw(barrier)
    # плечи: от шва рукава к внешнему контуру
    d.line([(300, 706), (432, 704)], fill=255, width=3)
    d.line([(705, 704), (840, 706)], fill=255, width=3)
    # шов рукава: подстраховка, чтобы рукав не «перетёк» в пиджак
    d.line([(428, 704), (418, 860), (405, 1112)], fill=255, width=2)
    d.line([(706, 704), (712, 860), (722, 1112)], fill=255, width=2)


def build(rgb):
    H, W = rgb.shape[:2]
    lum = rgb.mean(2)
    ink = lum < 0.38
    white = rgb.min(2) > 0.93
    lab_w, _ = ndimage.label(white)
    border = set(np.unique(np.r_[lab_w[0], lab_w[-1], lab_w[:, 0], lab_w[:, -1]])) - {0}
    bg = np.isin(lab_w, list(border))
    bar = Image.new("L", (W, H), 0)
    cuts(bar)
    barrier = np.asarray(bar) > 0
    region = ~ink & ~bg & ~barrier
    lab, n = ndimage.label(region)
    yy = np.arange(H)[:, None].repeat(W, 1)

    def comp(pts):
        m = np.zeros((H, W), bool)
        for x, y in pts:
            l = lab[y, x]
            if l:
                m |= lab == l
        return m

    arms = {}
    for s, pts in ARM_SEEDS.items():
        m = comp(pts)
        # линии вокруг областей руки тоже принадлежат руке
        m = ndimage.binary_dilation(m, iterations=10) & (ink | m | barrier | (lum < 0.85)) | m
        m &= ~bg
        arms[s] = m
    # голова: все области, начинающиеся выше подбородка и не уходящие ниже плеч
    head_reg = np.zeros((H, W), bool)
    objs = ndimage.find_objects(lab)
    for i, sl in enumerate(objs, 1):
        if sl is None:
            continue
        y0, y1 = sl[0].start, sl[0].stop
        if y0 < 600 and y1 < 690:
            head_reg |= lab == i
    head = (ndimage.binary_dilation(head_reg, iterations=10) & (ink | head_reg | (lum < 0.85))) | head_reg
    head &= yy < 690
    for s in arms:
        head &= ~arms[s]
    alpha = (~bg).astype(np.float32)
    return dict(head=head, arms=arms, bg=bg, ink=ink, alpha=alpha)


def body_layer(rgb, L):
    """Тело без головы и рук + дорисовка закрытых мест."""
    body = rgb.copy()
    a = L["alpha"].copy()
    rm = L["head"] | L["arms"]["L"] | L["arms"]["R"]
    a[rm] = 0
    # светлые пиксели сглаживания вокруг вырезанных рук — тоже прочь
    arms = L["arms"]["L"] | L["arms"]["R"]
    near = ndimage.binary_dilation(arms, iterations=4)
    a[near & (rgb.mean(2) > 0.78)] = 0
    # мелкие оторванные кусочки (пуговицы рукава и т.п.) — прочь
    labb, nb = ndimage.label(a > 0.5)
    if nb:
        sizes = ndimage.sum(np.ones_like(a), labb, range(1, nb + 1))
        small = np.isin(labb, np.nonzero(sizes < 1500)[0] + 1)
        a[small] = 0
    # контур торса по фактической границе выреза: край тела, граничащий с рукой, обводим линией
    solid = a > 0.5
    cut_edge = solid & ndimage.binary_dilation(arms & ~solid, iterations=3)
    cut_edge = ndimage.binary_dilation(cut_edge, iterations=1) & solid
    body[cut_edge] = (0.07, 0.07, 0.09)
    im = Image.fromarray((np.dstack([np.clip(body, 0, 1), a]) * 255).astype(np.uint8), "RGBA")
    d = ImageDraw.Draw(im)
    skin = (250, 232, 222, 255)
    line = (20, 20, 25, 255)
    # шея под головой
    d.rounded_rectangle((532, 560, 598, 660), 18, fill=skin)
    d.line([(533, 575), (536, 660)], fill=line, width=4)
    d.line([(597, 575), (594, 660)], fill=line, width=4)
    d.ellipse((545, 610, 585, 640), fill=(236, 210, 205, 255))   # тень под подбородком
    # плечевой край: короткая дуга, чтобы тело не выглядело обрезанным
    d.line([(432, 700), (470, 676)], fill=line, width=5)
    d.line([(702, 700), (664, 676)], fill=line, width=5)
    # воротник/ворот снова поверх шеи (берём из исходника область воротника)
    out = np.asarray(im, np.float32) / 255
    collar = (np.arange(rgb.shape[0])[:, None] >= 612) & (~L["head"]) & (~L["bg"])
    collar &= (np.arange(rgb.shape[1])[None, :] > 480) & (np.arange(rgb.shape[1])[None, :] < 650)
    collar &= np.arange(rgb.shape[0])[:, None] < 700
    out[collar, :3] = rgb[collar]
    out[collar, 3] = 1
    return out


def arm_with_cap(rgb, mask, side):
    """Рукав со скруглённым верхом — для поднятой руки (срез у плеча не виден)."""
    im = Image.fromarray((np.dstack([rgb, mask.astype(np.float32)]) * 255).astype(np.uint8), "RGBA")
    cap = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(cap)
    cx, cy, r = (388, 716, 44) if side == "L" else (748, 716, 44)
    d.pieslice((cx - r, cy - r, cx + r, cy + r), 180, 360, fill=(158, 159, 164, 255))
    d.arc((cx - r, cy - r, cx + r, cy + r), 180, 360, fill=(15, 15, 20, 255), width=5)
    cap.alpha_composite(im)
    return cap


def layer(rgb, mask):
    return np.dstack([rgb, mask.astype(np.float32)])
