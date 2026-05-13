"""Operator assembly and block conversion utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .spaces import Array, SectorDecomposition


def _as_complex(matrix: NDArray[np.complexfloating] | NDArray[np.floating]) -> Array:
    return np.asarray(matrix, dtype=np.complex128)


@dataclass(frozen=True)
class BlockOperator:
    """Block representation of ``A`` relative to ``H = hidden direct-sum observable``."""

    A_pp: Array
    A_pa: Array
    A_ap: Array
    A_aa: Array

    @property
    def hidden_dim(self) -> int:
        return int(self.A_pp.shape[0])

    @property
    def observable_dim(self) -> int:
        return int(self.A_aa.shape[0])

    @property
    def n(self) -> int:
        return self.hidden_dim + self.observable_dim

    def as_matrix(self) -> Array:
        return assemble_block_operator(self.A_pp, self.A_pa, self.A_ap, self.A_aa)


def assemble_block_operator(
    A_pp: NDArray[np.complexfloating] | NDArray[np.floating],
    A_pa: NDArray[np.complexfloating] | NDArray[np.floating],
    A_ap: NDArray[np.complexfloating] | NDArray[np.floating],
    A_aa: NDArray[np.complexfloating] | NDArray[np.floating],
) -> Array:
    """Assemble the full block matrix ``[[A_pp, A_pa], [A_ap, A_aa]]``."""

    pp = _as_complex(A_pp)
    pa = _as_complex(A_pa)
    ap = _as_complex(A_ap)
    aa = _as_complex(A_aa)
    h = pp.shape[0]
    a = aa.shape[0]
    expected = {
        "A_pp": (h, h),
        "A_pa": (h, a),
        "A_ap": (a, h),
        "A_aa": (a, a),
    }
    actual = {"A_pp": pp.shape, "A_pa": pa.shape, "A_ap": ap.shape, "A_aa": aa.shape}
    for name, shape in expected.items():
        if actual[name] != shape:
            raise ValueError(f"{name} has shape {actual[name]}, expected {shape}")
    return np.block([[pp, pa], [ap, aa]]).astype(np.complex128)


def split_block_operator(A: Array, hidden_dim: int) -> BlockOperator:
    """Split a block-coordinate matrix into hidden/observable blocks."""

    A = _as_complex(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")
    if not 0 < hidden_dim < A.shape[0]:
        raise ValueError("hidden_dim must lie strictly between 0 and n")
    h = hidden_dim
    return BlockOperator(
        A_pp=A[:h, :h].copy(),
        A_pa=A[:h, h:].copy(),
        A_ap=A[h:, :h].copy(),
        A_aa=A[h:, h:].copy(),
    )


def block_matrix_from_full(A: Array, sector: SectorDecomposition) -> Array:
    """Represent an ambient matrix in sector coordinates."""

    A = _as_complex(A)
    U = sector.basis_matrix
    return U.conj().T @ A @ U


def block_operator_from_full(A: Array, sector: SectorDecomposition) -> BlockOperator:
    """Return the four sector blocks of an ambient matrix."""

    return split_block_operator(block_matrix_from_full(A, sector), sector.hidden_dim)


def full_matrix_from_block_operator(blocks: BlockOperator, sector: SectorDecomposition) -> Array:
    """Convert a sector-coordinate block operator into ambient coordinates."""

    if blocks.n != sector.n:
        raise ValueError("block dimension does not match sector dimension")
    U = sector.basis_matrix
    return U @ blocks.as_matrix() @ U.conj().T


def leakage_full(A: Array, P: Array, Q: Array) -> Array:
    """Return the ambient leakage operator ``Q A P``."""

    return _as_complex(Q) @ _as_complex(A) @ _as_complex(P)


def leakage_block(A: Array, sector: SectorDecomposition) -> Array:
    """Return the coordinate matrix for ``A_ap : hidden -> observable``."""

    return block_operator_from_full(A, sector).A_ap

