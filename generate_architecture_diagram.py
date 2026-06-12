"""
Generates the model architecture diagram for the DR-XAI report.
Output: reports/architecture_diagram.png  (300 dpi, white background)

Run:  venv/bin/python generate_architecture_diagram.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
IDRID_DIR = 'data/raw/idrid/B. Disease Grading/1. Original Images/a. Training Set/'
OUT_DIR   = 'reports'
os.makedirs(OUT_DIR, exist_ok=True)

# Pick one image per grade level for visual variety (Grade 0, 2, 4)
SAMPLE_FILES = {
    'Grade 0\n(No DR)':           'IDRiD_001.jpg',
    'Grade 2\n(Moderate DR)':     'IDRiD_016.jpg',
    'Grade 4\n(Proliferative DR)':'IDRiD_007.jpg',
}

# ── Colour palette ────────────────────────────────────────────────────────────
C_BACKBONE  = ('#F0E8FF', '#9B6DFF')   # purple tint, purple border
C_POOL      = ('#E8F4FF', '#4F8FFF')   # blue tint
C_HEAD      = ('#E8FFF5', '#00C97A')   # green tint
C_OUTPUT    = {0:'#22C55E', 1:'#84CC16', 2:'#F59E0B', 3:'#F97316', 4:'#EF4444'}
C_ARROW     = '#555566'
C_HEADER    = '#2D1B4E'

# ── Figure ────────────────────────────────────────────────────────────────────
FW, FH = 22, 9
fig = plt.figure(figsize=(FW, FH), facecolor='white')
ax  = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, FW)
ax.set_ylim(0, FH)
ax.axis('off')

# ── Helper: rounded box ───────────────────────────────────────────────────────
def rbox(x, y, w, h, fc, ec, lw=2.0, radius=0.3):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0",
        facecolor=fc, edgecolor=ec, linewidth=lw,
        zorder=2
    )
    ax.add_patch(p)

def label(x, y, txt, size=10, weight='bold', color='#1a1a2e', ha='center', va='center'):
    ax.text(x, y, txt, fontsize=size, fontweight=weight,
            color=color, ha=ha, va=va, zorder=3,
            fontfamily='DejaVu Sans')

def arrow(x1, y1, x2, y2, color=C_ARROW, lw=2.5):
    ax.annotate(
        '', xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle='->', color=color,
            lw=lw, mutation_scale=18
        ),
        zorder=4
    )

# ═══════════════════════════════════════════════════════════════════════════════
# ZONE 1 — Input images  (x: 0.4 – 3.2)
# ═══════════════════════════════════════════════════════════════════════════════
IMG_W, IMG_H = 2.2, 2.0
img_ys = [6.0, 3.5, 1.0]   # top, middle, bottom centre-y

# outer label
label(1.7, 8.65, 'Input Images', size=11, color=C_HEADER)

for i, (grade_lbl, fname) in enumerate(SAMPLE_FILES.items()):
    cx, cy = 1.7, img_ys[i]
    ix, iy = cx - IMG_W/2, cy - IMG_H/2

    fpath = os.path.join(IDRID_DIR, fname)
    if os.path.exists(fpath):
        try:
            img = Image.open(fpath).convert('RGB')
            img.thumbnail((224, 224))
            img_arr = np.array(img)
            ax.imshow(img_arr,
                      extent=[ix, ix+IMG_W, iy, iy+IMG_H],
                      aspect='auto', zorder=2)
        except Exception:
            rbox(ix, iy, IMG_W, IMG_H, '#cccccc', '#888888')
    else:
        # placeholder gradient if image missing
        grad = np.linspace(0.3, 0.7, 50).reshape(1, -1) * np.ones((50, 1))
        ax.imshow(grad, extent=[ix, ix+IMG_W, iy, iy+IMG_H],
                  cmap='RdYlGn', aspect='auto', zorder=2, vmin=0, vmax=1)

    # border around image
    rbox(ix, iy, IMG_W, IMG_H, 'none', '#888899', lw=1.5)

    # grade caption below
    label(cx, iy - 0.28, grade_lbl, size=8.5, weight='normal', color='#333344')

    # arrow from image to backbone
    arrow(ix + IMG_W, cy, 4.1, cy)

# ═══════════════════════════════════════════════════════════════════════════════
# ZONE 2 — Backbone: EfficientNet-B4  (x: 4.1 – 7.1)
# ═══════════════════════════════════════════════════════════════════════════════
BX, BY, BW, BH = 4.1, 0.7, 3.0, 7.6
rbox(BX, BY, BW, BH, C_BACKBONE[0], C_BACKBONE[1], lw=2.5)
label(BX + BW/2, BY + BH + 0.3, 'Backbone Network', size=11, color=C_HEADER)

# 3D box illusion for EfficientNet
EX, EY, EW, EH = 4.5, 2.9, 2.2, 2.4
DEPTH = 0.32

# draw 3 faces of the 3-D box
faces = [
    # front face
    dict(xy=(EX,       EY),       w=EW,     h=EH,   fc='#DDD0FF', ec=C_BACKBONE[1]),
    # top face (parallelogram approximated as shifted rect)
    dict(xy=(EX+DEPTH, EY+EH),    w=EW,     h=DEPTH, fc='#C8B8F0', ec=C_BACKBONE[1]),
    # right face
    dict(xy=(EX+EW,    EY+DEPTH), w=DEPTH,  h=EH,   fc='#B8A8E0', ec=C_BACKBONE[1]),
]
for f in faces:
    rbox(f['xy'][0], f['xy'][1], f['w'], f['h'], f['fc'], f['ec'], lw=1.5)

label(EX + EW/2, EY + EH/2 + 0.15, 'EfficientNet-B4', size=10.5, color='#2D1B4E')
label(EX + EW/2, EY + EH/2 - 0.25, 'pretrained · ImageNet', size=8,
      weight='normal', color='#5540AA')

# annotation inside backbone box
label(BX + BW/2, BY + 1.2, '448 channels\n(MBConv blocks 1–7)', size=8,
      weight='normal', color='#665599')

# arrow out of backbone
arrow(BX + BW, 4.5, 8.0, 4.5)

# ═══════════════════════════════════════════════════════════════════════════════
# ZONE 3 — Global Average Pooling  (x: 8.0 – 10.5)
# ═══════════════════════════════════════════════════════════════════════════════
GX, GY, GW, GH = 8.0, 3.3, 2.5, 2.4
rbox(GX, GY, GW, GH, C_POOL[0], C_POOL[1], lw=2.0)
label(GX + GW/2, GY + GH + 0.3, 'Feature\nExtraction', size=11, color=C_HEADER)
label(GX + GW/2, GY + GH/2 + 0.30, 'Global Avg Pool', size=9.5, color='#1A3A6E')
label(GX + GW/2, GY + GH/2 - 0.10, '↓', size=14, color='#4F8FFF', weight='normal')
label(GX + GW/2, GY + GH/2 - 0.50, '1792 features', size=9, color='#1A3A6E', weight='normal')

arrow(GX + GW, 4.5, 11.3, 4.5)

# ═══════════════════════════════════════════════════════════════════════════════
# ZONE 4 — Classifier Head  (x: 11.3 – 14.8)
# ═══════════════════════════════════════════════════════════════════════════════
HX, HY, HW, HH = 11.3, 1.2, 3.5, 6.6
rbox(HX, HY, HW, HH, C_HEAD[0], C_HEAD[1], lw=2.5)
label(HX + HW/2, HY + HH + 0.3, 'Classifier Head', size=11, color=C_HEADER)

# --- Dropout node ---
DX, DY = HX + HW/2, HY + HH - 1.15
circle_d = plt.Circle((DX, DY), 0.42, color='#B2F0D8', ec='#00C97A', lw=1.8, zorder=3)
ax.add_patch(circle_d)
label(DX, DY + 0.03, 'Dropout', size=8.5, color='#006644')
label(DX, DY - 0.26, '(p = 0.3)', size=7.5, weight='normal', color='#007755')

ax.annotate('', xy=(DX, DY - 0.42 - 0.1),
            xytext=(DX, DY - 0.42),
            arrowprops=dict(arrowstyle='->', color='#00C97A', lw=1.8))

# --- FC Layer nodes (MLP style) ---
N_IN, N_OUT = 5, 5
NODE_R = 0.22
x_in  = HX + 0.9
x_out = HX + HW - 0.9
ys_in  = np.linspace(HY + 0.6, DY - 0.85, N_IN)
ys_out = np.linspace(HY + 0.6, DY - 0.85, N_OUT)

# connections
for yi in ys_in:
    for yo in ys_out:
        ax.plot([x_in + NODE_R, x_out - NODE_R], [yi, yo],
                color='#AADDCC', lw=0.7, zorder=2, alpha=0.7)

# input nodes
for yi in ys_in:
    c = plt.Circle((x_in, yi), NODE_R, color='#C8F0E0', ec='#00C97A', lw=1.4, zorder=3)
    ax.add_patch(c)

# output nodes
for yo in ys_out:
    c = plt.Circle((x_out, yo), NODE_R, color='#A8E8D0', ec='#00C97A', lw=1.4, zorder=3)
    ax.add_patch(c)

# label between nodes
mid_y = (ys_in[0] + ys_in[-1]) / 2
label(HX + HW/2, mid_y + 0.05, 'FC  1792 → 5', size=9, color='#005533')
label(HX + HW/2, mid_y - 0.30, 'Linear layer', size=7.5, weight='normal', color='#007744')

# Softmax tag
label(HX + HW/2, HY + 0.28, 'Softmax', size=9.5, color='#005533')

arrow(HX + HW, 4.5, 15.6, 4.5)

# ═══════════════════════════════════════════════════════════════════════════════
# ZONE 5 — Output  (x: 15.6 – 21.6)
# ═══════════════════════════════════════════════════════════════════════════════
label(18.3, 8.65, 'Classification\nResult', size=11, color=C_HEADER)

GRADE_NAMES = ['Grade 0\nNo DR', 'Grade 1\nMild DR',
               'Grade 2\nModerate DR', 'Grade 3\nSevere DR',
               'Grade 4\nProliferative DR']
OBX, OBW, OBH = 15.6, 2.5, 1.25
gaps = np.linspace(0.9, 7.4, 5)

for i, (g, col) in enumerate(C_OUTPUT.items()):
    oy = gaps[i] - OBH/2
    # coloured slab
    rbox(OBX, oy, OBW, OBH, col + '22', col, lw=2.0)
    label(OBX + OBW/2, oy + OBH/2 + 0.08,
          GRADE_NAMES[i].split('\n')[0], size=9.5, color=col)
    label(OBX + OBW/2, oy + OBH/2 - 0.22,
          GRADE_NAMES[i].split('\n')[1], size=8, weight='normal', color='#333')

# connecting lines from arrow tip to each output box
for i in range(5):
    oy  = gaps[i]
    ax.plot([15.6, 15.6], [oy, 4.5], color='#888899', lw=1.2,
            linestyle='--', zorder=1, alpha=0.6)

# ═══════════════════════════════════════════════════════════════════════════════
# Title
# ═══════════════════════════════════════════════════════════════════════════════
label(FW/2, 0.38,
      'DR·XAI — EfficientNet-B4 Architecture for Diabetic Retinopathy Grading',
      size=12, color='#1a1a2e', weight='bold')

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = os.path.join(OUT_DIR, 'architecture_diagram.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print(f"Saved → {out_path}")
plt.close()
