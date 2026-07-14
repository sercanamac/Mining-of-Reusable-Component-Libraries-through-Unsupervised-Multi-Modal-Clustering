"""Single staircase, all 9 viewpoints, 3x3 grid — for the vision-pipeline slide."""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.image import imread

sys.path.insert(0, 'notebooks/data_analysis/feature_engineering')
sys.path.insert(0, 'notebooks/data_analysis/experiments')

from _common import PROJECT, load_data

RENDERS_DIR = PROJECT / 'processed_data/rendersv2/colored'
VIEWPOINTS = ['top', 'front', 'right',
              'back', 'iso', 'left',
              'front_right', 'back_left', 'bottom']

# Find any staircase mesh that has all 9 views available
df, y = load_data()
import numpy as np
stair_mask = np.array([yi == 'Stair' for yi in y])
stair_gids = df['GlobalId'].values[stair_mask].tolist()
chosen = None
for gid in stair_gids:
    rd = RENDERS_DIR / gid
    if all((rd / f'{v}.png').exists() for v in VIEWPOINTS):
        chosen = gid
        break
if chosen is None:
    print('no staircase with all 9 viewpoints found')
    sys.exit(1)
print(f'using staircase: {chosen}')

fig, axes = plt.subplots(3, 3, figsize=(10, 10))
for ax, vp in zip(axes.flat, VIEWPOINTS):
    img = imread(RENDERS_DIR / chosen / f'{vp}.png')
    ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(vp, fontsize=12, color='#333', labelpad=4)
    for s in ax.spines.values():
        s.set_edgecolor('#ddd'); s.set_linewidth(0.5)

fig.suptitle('One object, 9 viewpoints', fontsize=16, y=0.96)
fig.tight_layout(rect=(0, 0, 1, 0.94))
out = 'notebooks/data_analysis/presentation/figures/17_staircase_9views.png'
fig.savefig(out, dpi=160, bbox_inches='tight', facecolor='white')
print(f'saved → {out}')
