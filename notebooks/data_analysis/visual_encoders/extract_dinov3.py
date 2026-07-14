"""Extract DINOv3 ViT-L/16 visual embeddings on IFCNetCore.

For each of the 7930 IFCNet objects, loads all 12 stock renders, runs them
through DINOv3 ViT-L/16, mean-pools the CLS token across views, and writes
one .npy per object (1024-d float32).

Output layout (mirrors the existing _loaders.py convention so
`load_visual(..., 'dinov3', 'colorless', spec=IFCNET)` finds it):
    data/IFCNetCore/processed/rendered_features/dinov3/colorless/{obj_id}.npy

Cluster setup (one-time):
    git clone https://github.com/facebookresearch/dinov3.git
    wget -O dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
        https://huggingface.co/jaychempan/dinov3/resolve/<commit>/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
    # NOTE: the hubconf REQUIRES the filename to match the expected hash pattern;
    # do NOT rename the weights file to e.g. `model.pth` — it will raise
    # `ValueError: Unexpected weights specification for the ViT-L backbone`.

Run:
    python extract_dinov3.py \
        --dinov3-repo /path/to/dinov3 \
        --weights /path/to/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
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

# DINOv3 ViT-L/16 conventions (from the official tutorial notebook)
PATCH_SIZE = 16
IMAGE_SIZE = 768                    # 48 × 48 = 2304 patches + CLS
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
ENCODER_DIR = 'dinov3'


def resize_for_dinov3(pil, image_size: int = IMAGE_SIZE, patch_size: int = PATCH_SIZE):
    """Resize so each dim is a multiple of patch_size, then ToTensor + normalize."""
    import torchvision.transforms.functional as TF
    w, h = pil.size
    h_patches = int(image_size / patch_size)
    w_patches = int((w * image_size) / (h * patch_size))
    resized = TF.resize(pil, (h_patches * patch_size, w_patches * patch_size))
    tens = TF.to_tensor(resized)
    return TF.normalize(tens, mean=IMAGENET_MEAN, std=IMAGENET_STD)


def load_dinov3_model(dinov3_repo: Path, weights: Path, device: str):
    """torch.hub.load DINOv3 ViT-L/16 from a local repo clone + downloaded weights."""
    import torch
    model = torch.hub.load(
        str(dinov3_repo), 'dinov3_vitl16',
        source='local',
        weights=str(weights),
        skip_validation=True,
    )
    model = model.to(device).eval()
    return model


def encode_object(model, views, device):
    """Stack all views → batch forward → mean over views → (1024,) float32."""
    import torch
    import numpy as np
    from PIL import Image
    with torch.no_grad():
        tensors = []
        for v in views:
            pil = Image.open(v).convert('RGB')
            tensors.append(resize_for_dinov3(pil).unsqueeze(0))
        batch = torch.cat(tensors).to(device, non_blocking=True)  # (N_views, 3, H, W)
        feats = model(batch)                                      # (N_views, 1024)
        return feats.mean(dim=0).cpu().numpy().astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    md, rd, od = default_paths('ifcnet')
    p.add_argument('--metadata', type=Path, default=md,
                   help='Path to metadata.json (default: data/IFCNetCore/metadata.json)')
    p.add_argument('--renders-root', type=Path, default=rd,
                   help='Root of unzipped PNG renders')
    p.add_argument('--output-root', type=Path, default=od,
                   help='Root for rendered_features/<encoder>/<variant>/...')
    p.add_argument('--variant', default='colorless',
                   choices=['colored', 'colorless'],
                   help='Subdir under the encoder name (IFCNet renders are R=G=B so '
                        '"colorless" matches existing Bentley convention; '
                        'use "colored" only if you swap in different renders)')
    p.add_argument('--dinov3-repo', type=Path, required=True,
                   help='Local clone of github.com/facebookresearch/dinov3')
    p.add_argument('--weights', type=Path, required=True,
                   help='Path to dinov3_vitl16_pretrain_lvd1689m-*.pth')
    p.add_argument('--device', default='cuda',
                   help='cuda | cuda:0 | cpu (default: cuda)')
    p.add_argument('--limit', type=int, default=None,
                   help='Process only the first N objects (for smoke tests)')
    args = p.parse_args()

    out_dir = resolve_output_dir(args.output_root, ENCODER_DIR, args.variant)
    print(f'[dinov3] output → {out_dir}', flush=True)

    entries = load_metadata(args.metadata, project_root())
    print(f'[dinov3] loaded metadata: {len(entries)} objects', flush=True)
    if args.limit:
        entries = entries[:args.limit]

    pending = filter_pending(entries, out_dir)
    print(f'[dinov3] pending: {len(pending)} / {len(entries)} '
          f'({len(entries) - len(pending)} already done)', flush=True)
    if not pending:
        print('[dinov3] nothing to do — output is up to date')
        return

    import numpy as np   # deferred — keeps --help usable without torch installed
    model = load_dinov3_model(args.dinov3_repo, args.weights, args.device)
    print(f'[dinov3] model loaded on {args.device}', flush=True)

    for e in iter_with_progress(pending, 'dinov3'):
        if not e.views:
            print(f'  [warn] no views for {e.obj_id}, skipping', flush=True)
            continue
        try:
            vec = encode_object(model, e.views, args.device)
        except Exception as exc:
            print(f'  [error] {e.obj_id}: {exc}', flush=True)
            continue
        np.save(out_dir / f'{e.obj_id}.npy', vec)


if __name__ == '__main__':
    main()
