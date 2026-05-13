"""Linear time evolution and Duhamel reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad_vec, solve_ivp
from scipy.linalg import expm

from .spaces import Array, SectorDecomposition
from .utils import row_norms


@dataclass(frozen=True)
class LinearTrajectory:
    """State trajectory together with hidden/observable projections."""

    t: NDArray[np.float64]
    psi: Array
    psi_p: Array
    psi_a: Array
    psi_p_coords: Array
    psi_a_coords: Array


def exact_linear_flow(A: Array, psi0: Array, t_eval: NDArray[np.floating]) -> Array:
    """Evaluate ``exp(t A) psi0`` exactly with finite-dimensional matrix exponentials."""

    A = np.asarray(A, dtype=np.complex128)
    psi0 = np.asarray(psi0, dtype=np.complex128)
    return np.vstack([expm(float(t) * A) @ psi0 for t in np.asarray(t_eval, dtype=float)])


def simulate_linear_ode(
    A: Array,
    psi0: Array,
    t_eval: NDArray[np.floating],
    *,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    method: str = "DOP853",
) -> Array:
    """Integrate ``d psi / dt = A psi`` with ``solve_ivp``."""

    A = np.asarray(A, dtype=np.complex128)
    psi0 = np.asarray(psi0, dtype=np.complex128)
    t_eval = np.asarray(t_eval, dtype=float)

    def rhs(_t: float, y: Array) -> Array:
        return A @ y

    sol = solve_ivp(rhs, (float(t_eval[0]), float(t_eval[-1])), psi0, t_eval=t_eval, rtol=rtol, atol=atol, method=method)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y.T.astype(np.complex128)


def make_linear_trajectory(
    A: Array,
    psi0: Array,
    t_eval: NDArray[np.floating],
    sector: SectorDecomposition,
    *,
    method: str = "exact",
) -> LinearTrajectory:
    """Compute a linear trajectory and decompose it into sector components."""

    if method == "exact":
        psi = exact_linear_flow(A, psi0, t_eval)
    elif method == "ode":
        psi = simulate_linear_ode(A, psi0, t_eval)
    else:
        raise ValueError("method must be 'exact' or 'ode'")
    psi_p = sector.project_hidden(psi)
    psi_a = sector.project_observable(psi)
    return LinearTrajectory(
        t=np.asarray(t_eval, dtype=float),
        psi=psi,
        psi_p=psi_p,
        psi_a=psi_a,
        psi_p_coords=sector.hidden_coordinates(psi),
        psi_a_coords=sector.observable_coordinates(psi),
    )


def duhamel_reconstruction_samples(
    A_aa: Array,
    A_ap: Array,
    psi_p_samples: Array,
    t_eval: NDArray[np.floating],
) -> Array:
    """Approximate the Duhamel integral from sampled hidden coordinates.

    The integration is trapezoidal on the supplied grid.  For high-accuracy
    theorem checks, prefer :func:`duhamel_reconstruction_quad`.
    """

    A_aa = np.asarray(A_aa, dtype=np.complex128)
    A_ap = np.asarray(A_ap, dtype=np.complex128)
    psi_p_samples = np.asarray(psi_p_samples, dtype=np.complex128)
    t_eval = np.asarray(t_eval, dtype=float)
    observable_dim = A_aa.shape[0]
    out = np.zeros((len(t_eval), observable_dim), dtype=np.complex128)
    trapezoid = getattr(np, "trapezoid", np.trapz)
    for i, t in enumerate(t_eval):
        if i == 0:
            continue
        values = []
        for s, psi_p in zip(t_eval[: i + 1], psi_p_samples[: i + 1], strict=True):
            values.append(expm(float(t - s) * A_aa) @ (A_ap @ psi_p))
        out[i] = trapezoid(np.vstack(values), x=t_eval[: i + 1], axis=0)
    return out


def duhamel_reconstruction_quad(
    A_aa: Array,
    A_ap: Array,
    psi_p: Callable[[float], Array],
    t_eval: NDArray[np.floating],
    *,
    epsabs: float = 1e-11,
    epsrel: float = 1e-11,
) -> Array:
    """Evaluate the variation-of-constants integral with adaptive quadrature."""

    A_aa = np.asarray(A_aa, dtype=np.complex128)
    A_ap = np.asarray(A_ap, dtype=np.complex128)
    t_eval = np.asarray(t_eval, dtype=float)
    observable_dim = A_aa.shape[0]
    out = np.zeros((len(t_eval), observable_dim), dtype=np.complex128)
    for i, t in enumerate(t_eval):
        if np.isclose(t, 0.0):
            continue

        def integrand(s: float) -> Array:
            return expm(float(t - s) * A_aa) @ (A_ap @ np.asarray(psi_p(float(s)), dtype=np.complex128))

        value, _err = quad_vec(integrand, 0.0, float(t), epsabs=epsabs, epsrel=epsrel)
        out[i] = value
    return out


def sector_energies(psi_p: Array, psi_a: Array) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return ``E_p = 0.5 ||psi_p||^2`` and ``E_a = 0.5 ||psi_a||^2``."""

    return 0.5 * row_norms(psi_p) ** 2, 0.5 * row_norms(psi_a) ** 2


def first_order_prediction(A_ap: Array, psi0_p: Array, t_eval: NDArray[np.floating]) -> Array:
    """Return the leading observable term ``t A_ap psi0_p``."""

    leak = np.asarray(A_ap, dtype=np.complex128) @ np.asarray(psi0_p, dtype=np.complex128)
    return np.asarray(t_eval, dtype=float)[:, None] * leak[None, :]
