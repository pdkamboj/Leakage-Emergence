"""Numerical diagnostics aligned with the theorem statements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eig, norm, solve

from .spaces import Array, SectorDecomposition
from .utils import max_relative_error, row_norms


@dataclass(frozen=True)
class DuhamelError:
    max_abs: float
    max_rel: float


def exact_hiddenness_error(psi_a: Array) -> float:
    """Maximum observable-sector norm along a trajectory."""

    return float(np.max(row_norms(psi_a)))


def first_order_emergence_error(
    psi_a_coords: Array,
    t_eval: NDArray[np.floating],
    leakage_vector: Array,
    *,
    samples: int = 8,
) -> float:
    """Compare ``psi_a(t) / t`` with ``A_ap psi0`` over early nonzero times."""

    t = np.asarray(t_eval, dtype=float)
    psi_a = np.asarray(psi_a_coords, dtype=np.complex128)
    leak = np.asarray(leakage_vector, dtype=np.complex128)
    nonzero = np.flatnonzero(t > 0)
    chosen = nonzero[:samples]
    if len(chosen) == 0:
        raise ValueError("at least one positive time sample is required")
    scaled = psi_a[chosen] / t[chosen, None]
    return float(np.max(row_norms(scaled - leak)))


def duhamel_error(direct: Array, reconstructed: Array) -> DuhamelError:
    """Return absolute and relative trajectory errors for Duhamel reconstruction."""

    direct = np.asarray(direct, dtype=np.complex128)
    reconstructed = np.asarray(reconstructed, dtype=np.complex128)
    return DuhamelError(
        max_abs=float(np.max(row_norms(direct - reconstructed))),
        max_rel=max_relative_error(direct, reconstructed),
    )


def resolvent_leakage(A: Array, P: Array, Q: Array, lambda_value: complex) -> Array:
    """Compute ``Q (lambda I - A)^(-1) P``."""

    A = np.asarray(A, dtype=np.complex128)
    P = np.asarray(P, dtype=np.complex128)
    Q = np.asarray(Q, dtype=np.complex128)
    matrix = lambda_value * np.eye(A.shape[0], dtype=np.complex128) - A
    return Q @ solve(matrix, P, assume_a="gen")


def resolvent_leakage_norm(A: Array, P: Array, Q: Array, lambda_value: complex) -> float:
    """Spectral norm of the leakage resolvent."""

    return float(norm(resolvent_leakage(A, P, Q, lambda_value), ord=2))


def scan_resolvent_leakage(
    A: Array,
    P: Array,
    Q: Array,
    lambda_values: Iterable[complex],
) -> NDArray[np.float64]:
    """Scan leakage-resolvent norms across a grid, returning NaN at singular points."""

    norms: list[float] = []
    for lam in lambda_values:
        try:
            norms.append(resolvent_leakage_norm(A, P, Q, complex(lam)))
        except Exception:
            norms.append(float("nan"))
    return np.asarray(norms, dtype=float)


def classify_eigenvectors(A: Array, sector: SectorDecomposition, *, tol: float = 1e-9) -> list[dict[str, object]]:
    """Classify eigenvectors as hidden, observable, or mixed."""

    A = np.asarray(A, dtype=np.complex128)
    eigenvalues, eigenvectors = eig(A)
    records: list[dict[str, object]] = []
    for idx, lam in enumerate(eigenvalues):
        v = eigenvectors[:, idx]
        v_norm = np.linalg.norm(v)
        if v_norm == 0:
            continue
        v = v / v_norm
        hidden_norm = float(np.linalg.norm(sector.P @ v))
        observable_norm = float(np.linalg.norm(sector.Q @ v))
        if observable_norm <= tol:
            kind = "hidden"
        elif hidden_norm <= tol:
            kind = "observable"
        else:
            kind = "mixed"
        records.append(
            {
                "index": idx,
                "eigenvalue": complex(lam),
                "hidden_norm": hidden_norm,
                "observable_norm": observable_norm,
                "kind": kind,
                "residual": float(np.linalg.norm(A @ v - lam * v)),
            }
        )
    return records


def hidden_eigenmode_residual(A: Array, v: Array, sector: SectorDecomposition) -> dict[str, float]:
    """Diagnostics for a candidate hidden eigenmode."""

    A = np.asarray(A, dtype=np.complex128)
    v = np.asarray(v, dtype=np.complex128)
    Av = A @ v
    rayleigh = np.vdot(v, Av) / np.vdot(v, v)
    return {
        "hidden_norm": float(np.linalg.norm(sector.P @ v)),
        "observable_norm": float(np.linalg.norm(sector.Q @ v)),
        "eigen_residual": float(np.linalg.norm(Av - rayleigh * v)),
        "QAv_norm": float(np.linalg.norm(sector.Q @ Av)),
    }

