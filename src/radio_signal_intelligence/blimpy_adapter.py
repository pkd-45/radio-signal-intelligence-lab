from __future__ import annotations

from pathlib import Path

import numpy as np


def load_waterfall(path: str | Path, f_start_mhz: float | None = None, f_stop_mhz: float | None = None) -> np.ndarray:
    """Load a Breakthrough Listen .fil/.h5 waterfall through optional blimpy.

    This adapter is intentionally not exercised in the default synthetic CI.
    It keeps real-data ingestion separate from benchmark claims.
    """
    try:
        from blimpy import Waterfall
    except ImportError as exc:
        raise RuntimeError("Install optional dependencies with: pip install -e '.[bl]'") from exc

    kwargs: dict[str, float] = {}
    if f_start_mhz is not None:
        kwargs["f_start"] = f_start_mhz
    if f_stop_mhz is not None:
        kwargs["f_stop"] = f_stop_mhz
    waterfall = Waterfall(str(path), **kwargs)
    data = np.asarray(waterfall.data)
    data = np.squeeze(data)
    if data.ndim != 2:
        raise ValueError(f"Expected a 2-D time-frequency array after squeeze, got {data.shape}")
    return data.astype(np.float32)


def tile_pair(on: np.ndarray, off: np.ndarray, size: int = 64, stride: int = 64) -> list[np.ndarray]:
    if on.shape != off.shape or on.ndim != 2:
        raise ValueError("ON and OFF arrays must be matching 2-D arrays")
    pairs: list[np.ndarray] = []
    for t0 in range(0, on.shape[0] - size + 1, stride):
        for f0 in range(0, on.shape[1] - size + 1, stride):
            on_patch = on[t0 : t0 + size, f0 : f0 + size]
            off_patch = off[t0 : t0 + size, f0 : f0 + size]
            pair = np.stack([on_patch, off_patch]).astype(np.float32)
            med = np.median(pair, axis=(1, 2), keepdims=True)
            mad = np.median(np.abs(pair - med), axis=(1, 2), keepdims=True)
            pair = (pair - med) / np.maximum(1.4826 * mad, 1e-6)
            pairs.append(pair.astype(np.float32))
    return pairs
