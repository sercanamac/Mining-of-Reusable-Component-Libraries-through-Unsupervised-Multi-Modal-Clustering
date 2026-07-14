"""Extract DuoDuoCLIP visual embeddings on IFCNetCore.

Unlike DINOv3 / SigLIP, DuoDuoCLIP's `encode_image` accepts an array of
multi-view images and pools them internally (the model was trained to combine
views in its forward pass), so we DO NOT mean-pool afterwards — we just stack
the 12 views and call encode_image once per object.

Output layout:
    data/IFCNetCore/processed/rendered_features/duoduo/colorless/{obj_id}.npy

Cluster setup (one-time):
    git clone https://github.com/3dlg-hcvc/DuoduoCLIP.git
    pip install DuoduoCLIP/open_clip_mod/
    # The default checkpoint 'Four_1to6F_bs1600_LT6.ckpt' is auto-downloaded
    # by get_model() on first call (≈900 MB to ~/.cache).

Run:
    python extract_duoduo.py --duoduoclip-repo /path/to/DuoduoCLIP
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from _ifcnet_views import (
    default_paths, filter_pending, iter_with_progress,
    load_metadata, project_root, resolve_output_dir,
)

ENCODER_DIR = 'duoduo'
DEFAULT_CKPT = 'Four_1to6F_bs1600_LT6.ckpt'
INPUT_RESOLUTION = 224


def load_duoduoclip(duoduoclip_repo: Path, ckpt: str, device: str):
    """Import DuoduoCLIP from a local clone and load the multi-view CLIP wrapper."""
    sys.path.insert(0, str(duoduoclip_repo))
    from src.model.wrapper import get_model     # noqa: E402  -- repo-local import
    return get_model(ckpt, device=device)


def encode_object(model, views):
    """Stack views as (N, H, W, 3) uint8 → encode_image (pools internally) → (D,) float32.

    The DuoDuoCLIP wrapper expects raw RGB images as numpy uint8 (per the
    reference notebook); it does its own normalization.
    """
    import torch
    import numpy as np
    from PIL import Image
    with torch.no_grad():
        imgs = []
        for v in views:
            pil = Image.open(v).convert('RGB').resize((INPUT_RESOLUTION, INPUT_RESOLUTION))
            imgs.append(np.expand_dims(np.asarray(pil), 0))   # (1, H, W, 3)
        batch = np.concatenate(imgs, axis=0)                  # (N_views, H, W, 3)
        feats = model.encode_image(batch)                     # (D,) — multi-view pooled
        return feats.cpu().numpy().reshape(-1).astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    md, rd, od = default_paths('ifcnet')
    p.add_argument('--metadata', type=Path, default=md)
    p.add_argument('--renders-root', type=Path, default=rd)
    p.add_argument('--output-root', type=Path, default=od)
    p.add_argument('--variant', default='colorless',
                   choices=['colored', 'colorless'])
    p.add_argument('--duoduoclip-repo', type=Path, required=True,
                   help='Local clone of github.com/3dlg-hcvc/DuoduoCLIP')
    p.add_argument('--ckpt', default=DEFAULT_CKPT,
                   help=f'DuoDuoCLIP checkpoint name (default: {DEFAULT_CKPT})')
    p.add_argument('--device', default='cuda',
                   help='cuda | cuda:0 | cpu (default: cuda)')
    p.add_argument('--limit', type=int, default=None,
                   help='Process only the first N objects (for smoke tests)')
    args = p.parse_args()

    out_dir = resolve_output_dir(args.output_root, ENCODER_DIR, args.variant)
    print(f'[duoduo] ckpt={args.ckpt}', flush=True)
    print(f'[duoduo] output → {out_dir}', flush=True)

    entries = load_metadata(args.metadata, project_root())
    print(f'[duoduo] loaded metadata: {len(entries)} objects', flush=True)
    if args.limit:
        entries = entries[:args.limit]

    pending = filter_pending(entries, out_dir)
    print(f'[duoduo] pending: {len(pending)} / {len(entries)} '
          f'({len(entries) - len(pending)} already done)', flush=True)
    if not pending:
        print('[duoduo] nothing to do — output is up to date')
        return

    import numpy as np
    model = load_duoduoclip(args.duoduoclip_repo, args.ckpt, args.device)
    print(f'[duoduo] model loaded on {args.device}', flush=True)

    for e in iter_with_progress(pending, 'duoduo'):
        if not e.views:
            print(f'  [warn] no views for {e.obj_id}, skipping', flush=True)
            continue
        try:
            vec = encode_object(model, e.views)
        except Exception as exc:
            print(f'  [error] {e.obj_id}: {exc}', flush=True)
            continue
        np.save(out_dir / f'{e.obj_id}.npy', vec)


if __name__ == '__main__':
    main()
