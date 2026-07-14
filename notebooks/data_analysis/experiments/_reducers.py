"""Dimensionality-reducer helpers with on-disk caching.

Supports identity, PCA, and UMAP. UMAP fits are cached to disk because they
are the expensive step of the sweep (~5-15 s per fit on 1692-row blocks).

Standard reducer grids per modality are produced by `make_reducer_configs()`
— use these in P1/P2 scripts for consistency.
"""
from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PCA_DIMS = [8, 16, 32, 64, 128]
UMAP_DIMS = [4, 8, 16, 64]

UMAP_PARAMS_TEXTVIS = dict(n_neighbors=30, min_dist=0.0, metric='cosine', random_state=42)
UMAP_PARAMS_GEOFUS = dict(n_neighbors=30, min_dist=0.0, metric='euclidean', random_state=42)

_EXP_DIR = Path(__file__).resolve().parent
_CACHE_ROOT_DEFAULT = _EXP_DIR.parent / 'results' / 'midterm' / 'cache' / 'reducers'


@dataclass(frozen=True)
class ReducerConfig:
    name: str              # short name, e.g. 'identity', 'pca_32', 'umap_8'
    kind: str              # 'identity' | 'pca' | 'umap'
    n_components: Optional[int]
    umap_params: Optional[dict] = None   # only for UMAP; merged on top of defaults

    def cfg_hash(self) -> str:
        payload = {'kind': self.kind, 'n_components': self.n_components,
                   'umap_params': self.umap_params}
        blob = json.dumps(payload, sort_keys=True).encode()
        return hashlib.md5(blob).hexdigest()[:10]


def _array_hash(X: np.ndarray) -> str:
    h = hashlib.md5()
    h.update(str(X.shape).encode())
    h.update(str(X.dtype).encode())
    h.update(np.ascontiguousarray(X).tobytes())
    return h.hexdigest()[:12]


def make_reducer_configs(modality: str) -> list[ReducerConfig]:
    """Return the standard reducer grid for a modality.

    modality ∈ {'geo', 'text', 'visual', 'fusion_textvis', 'fusion_geofus'}
    """
    if modality == 'geo':
        return [ReducerConfig('identity', 'identity', None)]

    if modality in ('text', 'visual'):
        umap_params = UMAP_PARAMS_TEXTVIS
    elif modality in ('fusion_textvis',):
        umap_params = UMAP_PARAMS_TEXTVIS
    elif modality in ('fusion_geofus',):
        umap_params = UMAP_PARAMS_GEOFUS
    else:
        raise ValueError(f'unknown modality {modality!r}')

    cfgs = [ReducerConfig('identity', 'identity', None)]
    for k in PCA_DIMS:
        cfgs.append(ReducerConfig(f'pca_{k}', 'pca', k))
    for k in UMAP_DIMS:
        cfgs.append(ReducerConfig(f'umap_{k}', 'umap', k, dict(umap_params)))
    return cfgs


def fit_reducer(
    cfg: ReducerConfig,
    X: np.ndarray,
    *,
    modality_name: str = 'modality',
    cache_dir: Optional[Path] = None,
    use_cache: bool = True,
) -> tuple[np.ndarray, dict]:
    """Fit the reducer on X and return (X_reduced, meta).

    meta = {'cfg': asdict, 'var_ret': float|None, 'cache_hit': bool,
            'fit_seconds': float}
    """
    import time
    t0 = time.perf_counter()

    if cfg.kind == 'identity':
        return X.astype(np.float32, copy=False), {
            'cfg': asdict(cfg), 'var_ret': None, 'cache_hit': False,
            'fit_seconds': 0.0,
        }

    if cfg.kind == 'pca':
        Xr = PCA(n_components=cfg.n_components, random_state=42).fit(X)
        X_reduced = Xr.transform(X).astype(np.float32)
        var_ret = float(Xr.explained_variance_ratio_.sum())
        return X_reduced, {
            'cfg': asdict(cfg), 'var_ret': var_ret, 'cache_hit': False,
            'fit_seconds': time.perf_counter() - t0,
        }

    if cfg.kind == 'umap':
        cache_dir = Path(cache_dir or _CACHE_ROOT_DEFAULT)
        cache_dir.mkdir(parents=True, exist_ok=True)
        x_hash = _array_hash(X)
        fname = f'{modality_name}__{x_hash}__{cfg.name}__{cfg.cfg_hash()}.npy'
        cache_path = cache_dir / fname
        if use_cache and cache_path.exists():
            X_reduced = np.load(cache_path)
            return X_reduced, {
                'cfg': asdict(cfg), 'var_ret': None, 'cache_hit': True,
                'fit_seconds': time.perf_counter() - t0,
                'cache_path': str(cache_path),
            }
        import umap
        params = {**(cfg.umap_params or {}), 'n_components': cfg.n_components}
        Xr = umap.UMAP(**params).fit_transform(X).astype(np.float32)
        if use_cache:
            np.save(cache_path, Xr)
        return Xr, {
            'cfg': asdict(cfg), 'var_ret': None, 'cache_hit': False,
            'fit_seconds': time.perf_counter() - t0,
            'cache_path': str(cache_path),
        }

    raise ValueError(f'unknown reducer kind {cfg.kind!r}')


def std_scale(X: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(X).astype(np.float32)


def reducer_row_meta(cfg: ReducerConfig, var_ret: Optional[float]) -> dict:
    """Columns to inject into sweep CSV rows for this reducer config."""
    return {
        'reducer': cfg.name,
        'reducer_kind': cfg.kind,
        'reducer_dim': cfg.n_components if cfg.n_components is not None else -1,
        'reducer_var_ret': var_ret if var_ret is not None else float('nan'),
        'reducer_hash': cfg.cfg_hash(),
    }
