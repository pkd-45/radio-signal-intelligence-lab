from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

CLASS_NAMES = [
    "noise",
    "candidate_drift",
    "persistent_narrowband_rfi",
    "broadband_burst",
    "spectral_comb_rfi",
]


@dataclass(frozen=True)
class SampleTruth:
    class_id: int
    class_name: str
    snr: float
    drift_bins_per_step: float
    appears_off: bool
    group_id: int
    seed: int


def _normalise(x: np.ndarray) -> np.ndarray:
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    scale = max(1.4826 * mad, 1e-6)
    return ((x - med) / scale).astype(np.float32)


def _add_line(
    arr: np.ndarray,
    amplitude: float,
    f0: float,
    drift: float,
    width: float = 0.8,
) -> None:
    nt, nf = arr.shape
    freq = np.arange(nf, dtype=float)
    centre_t = (nt - 1) / 2
    for t in range(nt):
        centre = f0 + drift * (t - centre_t)
        arr[t] += amplitude * np.exp(-0.5 * ((freq - centre) / width) ** 2)


def _add_burst(arr: np.ndarray, amplitude: float, t0: float, time_width: float) -> None:
    nt, nf = arr.shape
    time = np.arange(nt, dtype=float)
    profile = np.exp(-0.5 * ((time - t0) / time_width) ** 2)
    spectral = 0.65 + 0.35 * np.cos(np.linspace(0, 3 * np.pi, nf)) ** 2
    arr += amplitude * profile[:, None] * spectral[None, :]


def _add_comb(arr: np.ndarray, amplitude: float, start: int, spacing: int) -> None:
    nt, nf = arr.shape
    freq = np.arange(nf, dtype=float)
    modulation = 0.85 + 0.15 * np.sin(np.linspace(0, 4 * np.pi, nt))
    for centre in range(start, nf, spacing):
        line = np.exp(-0.5 * ((freq - centre) / 0.7) ** 2)
        arr += amplitude * modulation[:, None] * line[None, :]


def make_pair(
    class_id: int,
    seed: int,
    image_size: int = 64,
    group_id: int = 0,
) -> tuple[np.ndarray, SampleTruth]:
    """Create an ON/OFF dynamic-spectrum pair with known truth.

    Axis 0 is time and axis 1 is frequency. Output has shape (2, time, frequency).
    The model sees ON and OFF as separate channels, plus derived metadata.
    """
    if class_id not in range(len(CLASS_NAMES)):
        raise ValueError(f"class_id must be 0..{len(CLASS_NAMES) - 1}")

    rng = np.random.default_rng(seed)
    on = rng.normal(0.0, 1.0, (image_size, image_size))
    off = rng.normal(0.0, 1.0, (image_size, image_size))

    snr = 0.0
    drift = 0.0
    appears_off = False

    if class_id == 1:
        snr = float(rng.uniform(5.0, 11.0))
        drift = float(rng.choice([-1.0, 1.0]) * rng.uniform(0.08, 0.28))
        margin = abs(drift) * image_size / 2 + 4
        f0 = float(rng.uniform(margin, image_size - margin))
        _add_line(on, snr, f0, drift, width=float(rng.uniform(0.65, 1.05)))
    elif class_id == 2:
        snr = float(rng.uniform(5.0, 11.0))
        drift = float(rng.uniform(-0.02, 0.02))
        f0 = float(rng.uniform(6, image_size - 6))
        _add_line(on, snr, f0, drift)
        _add_line(off, snr * float(rng.uniform(0.8, 1.1)), f0 + rng.uniform(-0.4, 0.4), drift)
        appears_off = True
    elif class_id == 3:
        snr = float(rng.uniform(3.5, 7.0))
        t0 = float(rng.uniform(10, image_size - 10))
        width = float(rng.uniform(1.2, 3.5))
        _add_burst(on, snr, t0, width)
        if rng.random() < 0.65:
            _add_burst(off, snr * float(rng.uniform(0.7, 1.1)), t0 + rng.uniform(-3, 3), width)
            appears_off = True
    elif class_id == 4:
        snr = float(rng.uniform(3.0, 6.0))
        spacing = int(rng.integers(6, 12))
        start = int(rng.integers(2, spacing + 2))
        _add_comb(on, snr, start, spacing)
        _add_comb(off, snr * float(rng.uniform(0.8, 1.1)), start, spacing)
        appears_off = True

    # Mild bandpass structure makes the benchmark less trivial without hiding truth.
    bandpass = 0.15 * np.sin(np.linspace(0, 2 * np.pi, image_size))[None, :]
    on += bandpass
    off += bandpass * float(rng.uniform(0.85, 1.15))

    pair = np.stack([_normalise(on), _normalise(off)], axis=0)
    truth = SampleTruth(
        class_id=class_id,
        class_name=CLASS_NAMES[class_id],
        snr=snr,
        drift_bins_per_step=drift,
        appears_off=appears_off,
        group_id=group_id,
        seed=seed,
    )
    return pair, truth


def estimate_metadata(pair: np.ndarray) -> np.ndarray:
    """Derive six transparent summary features from an ON/OFF pair."""
    if pair.ndim != 3 or pair.shape[0] != 2:
        raise ValueError("pair must have shape (2, time, frequency)")
    on, off = pair.astype(float)
    residual = on - off
    nt, _ = on.shape

    peak_on = float(np.max(on))
    peak_off = float(np.max(off))
    on_off_peak_ratio = float((peak_on + 1e-3) / (peak_off + 1e-3))

    threshold = float(np.median(residual) + 4.0 * max(np.std(residual), 1e-6))
    occupancy = float(np.mean(residual > threshold))

    ridge = np.argmax(residual, axis=1).astype(float)
    ridge_amp = residual[np.arange(nt), ridge.astype(int)]
    keep = ridge_amp > np.percentile(ridge_amp, 65)
    if np.count_nonzero(keep) >= 5:
        slope = float(np.polyfit(np.arange(nt)[keep], ridge[keep], 1)[0])
        ridge_coherence = float(np.mean(ridge_amp[keep]))
    else:
        slope = 0.0
        ridge_coherence = 0.0

    time_power = np.maximum(residual, 0).sum(axis=1)
    freq_power = np.maximum(residual, 0).sum(axis=0)
    time_concentration = float(np.max(time_power) / (np.mean(time_power) + 1e-6))
    freq_concentration = float(np.max(freq_power) / (np.mean(freq_power) + 1e-6))

    features = np.array(
        [
            peak_on,
            peak_off,
            np.log1p(max(on_off_peak_ratio, 0.0)),
            occupancy,
            slope,
            ridge_coherence + 0.1 * time_concentration + 0.1 * freq_concentration,
        ],
        dtype=np.float32,
    )
    return features
