"""Finite-dimensional hidden/observable Hilbert-space geometry.

The infinite-dimensional paper works with a Hilbert space ``H`` and an
observation map ``Pi`` whose kernel is the hidden sector.  In this package
``H = C^n`` with the standard Hermitian inner product.  Hidden and observable
sectors are represented by orthogonal projectors ``P`` and ``Q = I - P``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import null_space

Array = NDArray[np.complex128]


def _as_complex_matrix(matrix: NDArray[np.complexfloating] | NDArray[np.floating]) -> Array:
    return np.asarray(matrix, dtype=np.complex128)


def orthonormalize_columns(basis: NDArray[np.complexfloating] | NDArray[np.floating], *, tol: float = 1e-12) -> Array:
    """Return an orthonormal basis for the column span of ``basis``.

    Raises
    ------
    ValueError
        If the supplied columns are linearly dependent at the requested
        tolerance.
    """

    mat = _as_complex_matrix(basis)
    if mat.ndim != 2:
        raise ValueError("basis must be a two-dimensional array")
    q, r = np.linalg.qr(mat)
    diag = np.abs(np.diag(r))
    rank = int(np.sum(diag > tol))
    if rank != mat.shape[1]:
        raise ValueError(f"basis columns are not independent; rank={rank}, columns={mat.shape[1]}")
    return q[:, : mat.shape[1]]


def projector_from_basis(basis: NDArray[np.complexfloating] | NDArray[np.floating], *, tol: float = 1e-12) -> Array:
    """Build the orthogonal projector onto the span of the supplied columns."""

    u = orthonormalize_columns(basis, tol=tol)
    return u @ u.conj().T


def complementary_projector(P: Array) -> Array:
    """Return ``I - P`` for a square projector matrix."""

    P = _as_complex_matrix(P)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("P must be a square matrix")
    return np.eye(P.shape[0], dtype=np.complex128) - P


def projector_errors(P: Array, Q: Array | None = None) -> dict[str, float]:
    """Return numerical residuals for projector identities."""

    P = _as_complex_matrix(P)
    if Q is None:
        Q = complementary_projector(P)
    else:
        Q = _as_complex_matrix(Q)
    I = np.eye(P.shape[0], dtype=np.complex128)
    return {
        "P_idempotence": float(np.linalg.norm(P @ P - P, ord=2)),
        "Q_idempotence": float(np.linalg.norm(Q @ Q - Q, ord=2)),
        "P_self_adjoint": float(np.linalg.norm(P - P.conj().T, ord=2)),
        "Q_self_adjoint": float(np.linalg.norm(Q - Q.conj().T, ord=2)),
        "orthogonality": float(np.linalg.norm(P @ Q, ord=2)),
        "completeness": float(np.linalg.norm(P + Q - I, ord=2)),
    }


def verify_projectors(P: Array, Q: Array | None = None, *, tol: float = 1e-10) -> bool:
    """Check the orthogonal projector identities numerically."""

    return all(error <= tol for error in projector_errors(P, Q).values())


def apply_matrix(matrix: Array, states: Array) -> Array:
    """Apply a column-vector matrix to either one state or row-stacked states."""

    matrix = _as_complex_matrix(matrix)
    arr = np.asarray(states, dtype=np.complex128)
    if arr.ndim == 1:
        return matrix @ arr
    if arr.ndim == 2:
        return arr @ matrix.T
    raise ValueError("states must be a vector or a two-dimensional row stack")


@dataclass(frozen=True)
class SectorDecomposition:
    """Orthogonal decomposition ``C^n = P sector direct-sum W sector``.

    The basis matrices have orthonormal columns.  ``hidden_basis`` spans
    ``ker(Pi)`` and ``observable_basis`` spans its orthogonal complement.
    """

    hidden_basis: Array
    observable_basis: Array
    P: Array
    Q: Array

    @classmethod
    def canonical(cls, hidden_dim: int, observable_dim: int) -> "SectorDecomposition":
        """Create the coordinate split ``C^(h+a) = C^h direct-sum C^a``."""

        if hidden_dim <= 0 or observable_dim <= 0:
            raise ValueError("hidden_dim and observable_dim must be positive")
        n = hidden_dim + observable_dim
        ambient_basis = np.eye(n, dtype=np.complex128)
        hidden_basis = ambient_basis[:, :hidden_dim]
        observable_basis = ambient_basis[:, hidden_dim:]
        P = hidden_basis @ hidden_basis.conj().T
        Q = observable_basis @ observable_basis.conj().T
        return cls(hidden_basis=hidden_basis, observable_basis=observable_basis, P=P, Q=Q)

    @classmethod
    def from_hidden_basis(
        cls,
        hidden_basis: NDArray[np.complexfloating] | NDArray[np.floating],
        *,
        tol: float = 1e-12,
    ) -> "SectorDecomposition":
        """Create a sector decomposition from any independent hidden basis."""

        hidden = orthonormalize_columns(hidden_basis, tol=tol)
        observable = null_space(hidden.conj().T, rcond=tol).astype(np.complex128)
        if observable.shape[1] == 0:
            raise ValueError("observable complement is empty")
        P = hidden @ hidden.conj().T
        Q = observable @ observable.conj().T
        return cls(hidden_basis=hidden, observable_basis=observable, P=P, Q=Q)

    @property
    def n(self) -> int:
        return int(self.hidden_basis.shape[0])

    @property
    def hidden_dim(self) -> int:
        return int(self.hidden_basis.shape[1])

    @property
    def observable_dim(self) -> int:
        return int(self.observable_basis.shape[1])

    @property
    def basis_matrix(self) -> Array:
        """Unitary matrix whose columns are hidden coordinates then observable coordinates."""

        return np.column_stack([self.hidden_basis, self.observable_basis]).astype(np.complex128)

    def project_hidden(self, states: Array) -> Array:
        return apply_matrix(self.P, states)

    def project_observable(self, states: Array) -> Array:
        return apply_matrix(self.Q, states)

    def hidden_coordinates(self, states: Array) -> Array:
        arr = np.asarray(states, dtype=np.complex128)
        if arr.ndim == 1:
            return self.hidden_basis.conj().T @ arr
        return arr @ self.hidden_basis.conj()

    def observable_coordinates(self, states: Array) -> Array:
        arr = np.asarray(states, dtype=np.complex128)
        if arr.ndim == 1:
            return self.observable_basis.conj().T @ arr
        return arr @ self.observable_basis.conj()

    def combine(self, hidden_coordinates: Array, observable_coordinates: Array) -> Array:
        """Build ambient states from hidden and observable coordinates."""

        h = np.asarray(hidden_coordinates, dtype=np.complex128)
        a = np.asarray(observable_coordinates, dtype=np.complex128)
        if h.ndim == 1 and a.ndim == 1:
            return self.hidden_basis @ h + self.observable_basis @ a
        if h.ndim == 2 and a.ndim == 2:
            return h @ self.hidden_basis.T + a @ self.observable_basis.T
        raise ValueError("hidden and observable coordinates must both be vectors or row stacks")
