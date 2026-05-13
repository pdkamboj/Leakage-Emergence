"""Small reproducibility and numerical helper functions."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.complex128]


def seeded_rng(seed: int = 20240512) -> np.random.Generator:
    """Return a NumPy generator with a fixed default seed."""

    return np.random.default_rng(seed)


def complex_normal(shape: int | tuple[int, ...], rng: np.random.Generator, *, scale: float = 1.0) -> Array:
    """Draw circular complex normal samples with variance controlled by ``scale``."""

    real = rng.normal(size=shape)
    imag = rng.normal(size=shape)
    return (scale / np.sqrt(2.0) * (real + 1j * imag)).astype(np.complex128)


def row_norms(values: NDArray[np.complexfloating] | NDArray[np.floating]) -> NDArray[np.float64]:
    """Euclidean norms for one vector or for row-stacked vectors."""

    arr = np.asarray(values)
    if arr.ndim == 1:
        return np.asarray(np.linalg.norm(arr), dtype=np.float64)
    return np.linalg.norm(arr, axis=1)


def max_relative_error(reference: NDArray[np.complexfloating], candidate: NDArray[np.complexfloating], *, floor: float = 1e-14) -> float:
    """Maximum row-wise relative error with a small denominator floor."""

    ref = np.asarray(reference)
    cand = np.asarray(candidate)
    diff = row_norms(ref - cand)
    denom = np.maximum(row_norms(ref), floor)
    return float(np.max(diff / denom))


def ensure_dir(path: str | Path) -> Path:
    """Create and return a directory path."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def finite_values(values: Iterable[float]) -> list[float]:
    """Return finite values from an iterable, useful for compact script summaries."""

    return [float(v) for v in values if np.isfinite(v)]


def estimate_loglog_slope(t: NDArray[np.floating], values: NDArray[np.floating]) -> float:
    """Least-squares slope of ``log(values)`` against ``log(t)`` for positive data."""

    t_arr = np.asarray(t, dtype=float)
    v_arr = np.asarray(values, dtype=float)
    mask = (t_arr > 0.0) & (v_arr > 0.0) & np.isfinite(t_arr) & np.isfinite(v_arr)
    if np.sum(mask) < 2:
        raise ValueError("at least two positive finite samples are required")
    coeffs = np.polyfit(np.log(t_arr[mask]), np.log(v_arr[mask]), deg=1)
    return float(coeffs[0])

