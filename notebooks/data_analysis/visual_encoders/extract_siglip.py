"""Extract SigLIP 2 visual embeddings on IFCNetCore.

For each object, load all 12 stock renders, run them through SigLIP 2
(`google/siglip2-large-patch16-512` by default), mean-pool the image
embeddings across views, and write one .npy per object.

Output dim: 1024 (SigLIP 2 large image projection).
Output layout (matches `_loaders.py` so `load_visual(..., 'siglip',
'colorless', spec=IFCNET)` resolves):
    data/IFCNetCore/processed/rendered_features/siglip2_large_16_512/colorless/{obj_id}.npy

Cluster setup:
    pip install "transformers>=4.49" torch pillow

Run:
    python extract_siglip.py
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

# Matches the existing _loaders.py mapping: VISUAL_ENCODERS['siglip'] = 'siglip2_large_16_512'
ENCODER_DIR = 'siglip2_large_16_512'
DEFAULT_MODEL_ID = 'google/siglip2-large-patch16-512'


def load_siglip(model_id: str, device: str):
    from transformers import AutoModel, AutoProcessor
    model = AutoModel.from_pretrained(model_id).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def encode_object(model, processor, views, device):
    """Run each view through the processor, batch them, mean-pool the image features."""
    import torch
    import numpy as np
    from PIL import Image
    with torch.no_grad():
        pil_list = [Image.open(v).convert('RGB') for v in views]
        inputs = processor(images=pil_list, padding=True, truncation=True,
                           return_tensors='pt')
        pixels = inputs['pixel_values'].to(device, non_blocking=True)  # (N_views, 3, H, W)
        feats = model.get_image_features(pixels)                       # (N_views, 1024)
        return feats.mean(dim=0).cpu().numpy().astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    md, rd, od = default_paths('ifcnet')
    p.add_argument('--metadata', type=Path, default=md)
    p.add_argument('--renders-root', type=Path, default=rd)
    p.add_argument('--output-root', type=Path, default=od)
    p.add_argument('--variant', default='colorless',
                   choices=['colored', 'colorless'])
    p.add_argument('--model-id', default=DEFAULT_MODEL_ID,
                   help=f'HuggingFace model id (default: {DEFAULT_MODEL_ID})')
    p.add_argument('--device', default='cuda',
                   help='cuda | cuda:0 | cpu (default: cuda)')
    p.add_argument('--limit', type=int, default=None,
                   help='Process only the first N objects (for smoke tests)')
    args = p.parse_args()

    out_dir = resolve_output_dir(args.output_root, ENCODER_DIR, args.variant)
    print(f'[siglip] model={args.model_id}', flush=True)
    print(f'[siglip] output → {out_dir}', flush=True)

    entries = load_metadata(args.metadata, project_root())
    print(f'[siglip] loaded metadata: {len(entries)} objects', flush=True)
    if args.limit:
        entries = entries[:args.limit]

    pending = filter_pending(entries, out_dir)
    print(f'[siglip] pending: {len(pending)} / {len(entries)} '
          f'({len(entries) - len(pending)} already done)', flush=True)
    if not pending:
        print('[siglip] nothing to do — output is up to date')
        return

    import numpy as np
    model, processor = load_siglip(args.model_id, args.device)
    print(f'[siglip] model loaded on {args.device}', flush=True)

    for e in iter_with_progress(pending, 'siglip'):
        if not e.views:
            print(f'  [warn] no views for {e.obj_id}, skipping', flush=True)
            continue
        try:
            vec = encode_object(model, processor, e.views, args.device)
        except Exception as exc:
            print(f'  [error] {e.obj_id}: {exc}', flush=True)
            continue
        np.save(out_dir / f'{e.obj_id}.npy', vec)


if __name__ == '__main__':
    main()
