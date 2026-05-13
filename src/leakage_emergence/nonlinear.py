"""Nonlinear finite-dimensional extensions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from .spaces import Array, SectorDecomposition

Nonlinearity = Callable[[Array], Array]


@dataclass(frozen=True)
class NonlinearTrajectory:
    """Nonlinear state trajectory with sector projections."""

    t: NDArray[np.float64]
    psi: Array
    psi_p: Array
    psi_a: Array
    psi_p_coords: Array
    psi_a_coords: Array


def cubic_self_interaction(lambda_nl: complex) -> Nonlinearity:
    """Elementwise cubic nonlinearity ``N(psi) = lambda |psi|^2 psi``."""

    coefficient = complex(lambda_nl)

    def nonlinearity(psi: Array) -> Array:
        psi = np.asarray(psi, dtype=np.complex128)
        return coefficient * np.abs(psi) ** 2 * psi

    return nonlinearity


def sector_preserving_cubic(
    sector: SectorDecomposition,
    *,
    lambda_hidden: complex,
    lambda_observable: complex | None = None,
) -> Nonlinearity:
    """Cubic response projected back into each sector separately.

    This is useful for constructing nonlinear examples where hiddenness is
    preserved by design: if ``psi`` is hidden, the observable component of
    ``N(psi)`` is exactly zero.
    """

    if lambda_observable is None:
        lambda_observable = lambda_hidden
    hidden_cubic = cubic_self_interaction(lambda_hidden)
    observable_cubic = cubic_self_interaction(lambda_observable)

    def nonlinearity(psi: Array) -> Array:
        psi = np.asarray(psi, dtype=np.complex128)
        p = sector.P @ psi
        a = sector.Q @ psi
        return sector.P @ hidden_cubic(p) + sector.Q @ observable_cubic(a)

    return nonlinearity


def nonlinear_vector_field(A: Array, nonlinearity: Nonlinearity) -> Callable[[float, Array], Array]:
    """Build ``f(t, psi) = A psi + N(psi)`` for ``solve_ivp``."""

    A = np.asarray(A, dtype=np.complex128)

    def rhs(_t: float, psi: Array) -> Array:
        psi = np.asarray(psi, dtype=np.complex128)
        return A @ psi + nonlinearity(psi)

    return rhs


def simulate_nonlinear(
    A: Array,
    nonlinearity: Nonlinearity,
    psi0: Array,
    t_eval: NDArray[np.floating],
    sector: SectorDecomposition,
    *,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    method: str = "DOP853",
) -> NonlinearTrajectory:
    """Integrate ``d psi / dt = A psi + N(psi)`` and decompose by sector."""

    psi0 = np.asarray(psi0, dtype=np.complex128)
    t_eval = np.asarray(t_eval, dtype=float)
    sol = solve_ivp(
        nonlinear_vector_field(A, nonlinearity),
        (float(t_eval[0]), float(t_eval[-1])),
        psi0,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        method=method,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    psi = sol.y.T.astype(np.complex128)
    psi_p = sector.project_hidden(psi)
    psi_a = sector.project_observable(psi)
    return NonlinearTrajectory(
        t=t_eval,
        psi=psi,
        psi_p=psi_p,
        psi_a=psi_a,
        psi_p_coords=sector.hidden_coordinates(psi),
        psi_a_coords=sector.observable_coordinates(psi),
    )


def hidden_vector_field_leakage(A: Array, nonlinearity: Nonlinearity, psi_hidden: Array, sector: SectorDecomposition) -> float:
    """Return ``||Q (A psi + N(psi))||`` for a hidden state."""

    psi_hidden = np.asarray(psi_hidden, dtype=np.complex128)
    vector = np.asarray(A, dtype=np.complex128) @ psi_hidden + nonlinearity(psi_hidden)
    return float(np.linalg.norm(sector.Q @ vector))

