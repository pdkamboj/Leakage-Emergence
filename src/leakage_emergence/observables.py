"""Observation maps, scalar observables, and threshold detection."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .spaces import Array, SectorDecomposition
from .utils import complex_normal, row_norms, seeded_rng


def canonical_awareness_map(sector: SectorDecomposition) -> Array:
    """Return ``Pi`` that extracts observable-sector coordinates.

    The kernel is exactly the hidden sector because ``Pi psi`` is the
    coordinate vector of ``Q psi`` in the observable basis.
    """

    return sector.observable_basis.conj().T.astype(np.complex128)


def general_awareness_map(
    sector: SectorDecomposition,
    *,
    output_dim: int | None = None,
    mixing: Array | None = None,
    seed: int = 20240512,
) -> Array:
    """Build a general linear map whose kernel is the hidden sector.

    The map has the form ``Pi = M U_a^*`` where ``U_a`` spans the observable
    sector and ``M`` has full column rank.  Therefore ``ker(Pi) = hidden``.
    """

    observable_dim = sector.observable_dim
    if mixing is None:
        if output_dim is None:
            output_dim = observable_dim
        if output_dim < observable_dim:
            raise ValueError("output_dim must be at least the observable dimension")
        rng = seeded_rng(seed)
        if output_dim == observable_dim:
            mixing = np.eye(observable_dim, dtype=np.complex128)
        else:
            for _ in range(100):
                candidate = complex_normal((output_dim, observable_dim), rng)
                if np.linalg.matrix_rank(candidate) == observable_dim:
                    mixing = candidate
                    break
            else:
                raise RuntimeError("failed to generate full-rank observable mixing")
    else:
        mixing = np.asarray(mixing, dtype=np.complex128)
        if mixing.ndim != 2 or mixing.shape[1] != observable_dim:
            raise ValueError("mixing must have shape (output_dim, observable_dim)")
        if np.linalg.matrix_rank(mixing) != observable_dim:
            raise ValueError("mixing must have full column rank")
    return np.asarray(mixing, dtype=np.complex128) @ sector.observable_basis.conj().T


def observe(Pi: Array, states: Array) -> Array:
    """Apply ``y = Pi psi`` to one state or row-stacked states."""

    Pi = np.asarray(Pi, dtype=np.complex128)
    arr = np.asarray(states, dtype=np.complex128)
    if arr.ndim == 1:
        return Pi @ arr
    if arr.ndim == 2:
        return arr @ Pi.T
    raise ValueError("states must be a vector or row-stacked matrix")


def observation_norms(Pi: Array, states: Array) -> NDArray[np.float64]:
    """Return ``||Pi psi(t)||`` for a trajectory."""

    return row_norms(observe(Pi, states))


def first_threshold_crossing(
    t: NDArray[np.floating],
    values: NDArray[np.floating],
    threshold: float,
) -> float:
    """Return the first threshold crossing time using linear interpolation.

    Returns ``np.inf`` if the threshold is never crossed on the sampled
    interval.
    """

    if threshold < 0:
        raise ValueError("threshold must be nonnegative")
    t_arr = np.asarray(t, dtype=float)
    v_arr = np.asarray(values, dtype=float)
    if t_arr.ndim != 1 or v_arr.ndim != 1 or len(t_arr) != len(v_arr):
        raise ValueError("t and values must be one-dimensional arrays of equal length")
    if np.any(np.diff(t_arr) < 0):
        raise ValueError("t must be nondecreasing")
    hits = np.flatnonzero(v_arr >= threshold)
    if len(hits) == 0:
        return float(np.inf)
    i = int(hits[0])
    if i == 0:
        return float(t_arr[0])
    v0, v1 = float(v_arr[i - 1]), float(v_arr[i])
    t0, t1 = float(t_arr[i - 1]), float(t_arr[i])
    if np.isclose(v1, v0):
        return t1
    alpha = (threshold - v0) / (v1 - v0)
    return float(t0 + alpha * (t1 - t0))


def scalar_observable(psi_a: Array, phi: Array) -> NDArray[np.float64]:
    """Return ``g_phi(t) = Re <psi_a(t), phi>``.

    The paper uses the convention that the Hilbert inner product is linear in
    its first argument, so this function computes ``sum psi_a * conj(phi)``.
    """

    arr = np.asarray(psi_a, dtype=np.complex128)
    phi = np.asarray(phi, dtype=np.complex128)
    if arr.ndim == 1:
        return np.asarray(float(np.real(np.sum(arr * np.conj(phi)))))
    return np.real(arr @ np.conj(phi))

