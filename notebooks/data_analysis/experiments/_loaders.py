"""Shared loaders for text and visual embeddings, aligned to a GlobalId order.

Every modality load function takes a target `gids` array (the canonical
evaluation order from `_common.load_data`) and returns a dense matrix X
plus an index mask `idx` such that `X` corresponds to `gids[idx]`.

Downstream sweep scripts always use `y[idx]` to keep labels aligned.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

_EXP_DIR = Path(__file__).resolve().parent
_PROJECT = _EXP_DIR.parents[2]
sys.path.insert(0, str(_EXP_DIR.parent / 'feature_engineering'))

from _common import ACTIVE_SPEC, DatasetSpec  # noqa: E402

PROCESSED = ACTIVE_SPEC.processed_root
GEMINI_ROOT = PROCESSED
VISUAL_ROOT = PROCESSED / 'rendered_features'

# ── Gemini text embedding paths ─────────────────────────────────────────────
# (version_id, folder_name under processed_data/)
TEXT_VERSIONS = {
    'v1':   'gemini_embeddings',
    'v2':   'gemini_embeddings_v2_geometry_full',
    'v3':   'gemini_embeddings_v2__v3_thinness',
    'v4':   'gemini_embeddings_v2_v4_compact',
    'v5':   'gemini_embeddings_v2_v5_natural',
    'v6':   'gemini_embeddings_v2_v6_keywords',
    'v6.1': 'gemini_embeddings_v2_v6_no_material',
    'v7':   'gemini_embeddings_v2_v7_ifc_aware',
    'gemma4': 'gemma4_embeddings',
}
TEXT_AGGREGATIONS = ['single', 'sum', 'concat', 'single_views']

# Per-version on-disk filename prefix. Defaults to 'gemini'.
_TEXT_EMBED_PREFIX = {
    'gemma4': 'gemma4',
}


def text_embed_dir(version: str, aggregation: str, spec: DatasetSpec = ACTIVE_SPEC) -> Path:
    """Path to the per-gid .npy directory for (version, aggregation)."""
    if version not in TEXT_VERSIONS:
        raise KeyError(f'unknown text version {version!r}; known: {list(TEXT_VERSIONS)}')
    if aggregation not in TEXT_AGGREGATIONS:
        raise KeyError(f'unknown aggregation {aggregation!r}; known: {TEXT_AGGREGATIONS}')
    prefix = _TEXT_EMBED_PREFIX.get(version, 'gemini')
    return spec.processed_root / TEXT_VERSIONS[version] / f'{prefix}_embed_{aggregation}'


def load_text(gids, version: str, aggregation: str = 'single', spec: DatasetSpec = ACTIVE_SPEC):
    """Load Gemini text embeddings for `gids` in order.

    Returns
    -------
    X : (n_found, d) float32 array
    idx : (n_found,) int array — indices into the input `gids` list that were
          successfully loaded
    """
    d = text_embed_dir(version, aggregation, spec)
    vs, idx = [], []
    for i, gid in enumerate(gids):
        p = d / f'{gid}.npy'
        if p.exists():
            vs.append(np.load(p).reshape(-1))
            idx.append(i)
    if not vs:
        raise RuntimeError(f'no text embeddings found under {d}')
    return np.stack(vs).astype(np.float32), np.array(idx)


# ── Visual embeddings ───────────────────────────────────────────────────────
VISUAL_ENCODERS = {
    'siglip':  'siglip2_large_16_512',
    'dinov3':  'dinov3',
    'duoduo':  'duoduo',
    'gemini':  'gemini',
}
VISUAL_COLOR_VARIANTS = ['colored', 'colorless']


def visual_embed_dir(encoder: str, variant: str, spec: DatasetSpec = ACTIVE_SPEC) -> Path:
    if encoder not in VISUAL_ENCODERS:
        raise KeyError(f'unknown visual encoder {encoder!r}; known: {list(VISUAL_ENCODERS)}')
    if variant not in VISUAL_COLOR_VARIANTS:
        raise KeyError(f'unknown variant {variant!r}; known: {VISUAL_COLOR_VARIANTS}')
    return spec.processed_root / 'rendered_features' / VISUAL_ENCODERS[encoder] / variant


def load_visual(gids, encoder: str, variant: str = 'colored', spec: DatasetSpec = ACTIVE_SPEC):
    d = visual_embed_dir(encoder, variant, spec)
    vs, idx = [], []
    for i, gid in enumerate(gids):
        p = d / f'{gid}.npy'
        if p.exists():
            vs.append(np.load(p).reshape(-1))
            idx.append(i)
    if not vs:
        raise RuntimeError(f'no visual embeddings found under {d}')
    return np.stack(vs).astype(np.float32), np.array(idx)


# ── Availability helpers ────────────────────────────────────────────────────
def available_text_configs() -> list[tuple[str, str, int]]:
    """Return (version, aggregation, file_count) for every non-empty directory."""
    out = []
    for v, folder in TEXT_VERSIONS.items():
        for a in TEXT_AGGREGATIONS:
            d = GEMINI_ROOT / folder / f'gemini_embed_{a}'
            if d.exists():
                n = sum(1 for _ in d.glob('*.npy'))
                if n > 0:
                    out.append((v, a, n))
    return out


def available_visual_configs() -> list[tuple[str, str, int]]:
    out = []
    for e, folder in VISUAL_ENCODERS.items():
        for var in VISUAL_COLOR_VARIANTS:
            d = VISUAL_ROOT / folder / var
            if d.exists():
                n = sum(1 for _ in d.glob('*.npy'))
                if n > 0:
                    out.append((e, var, n))
    return out
