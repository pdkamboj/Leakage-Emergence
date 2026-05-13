from __future__ import annotations

import numpy as np

from leakage_emergence.operators import (
    BlockOperator,
    block_operator_from_full,
    full_matrix_from_block_operator,
    leakage_full,
)
from leakage_emergence.spaces import SectorDecomposition


def test_block_operator_round_trip_in_rotated_sector() -> None:
    sector = SectorDecomposition.from_hidden_basis(np.array([[1.0], [1.0j], [0.25]], dtype=np.complex128))
    blocks = BlockOperator(
        A_pp=np.array([[-0.2 + 0.1j]], dtype=np.complex128),
        A_pa=np.array([[0.1, -0.05j]], dtype=np.complex128),
        A_ap=np.array([[0.3 - 0.1j], [-0.2j]], dtype=np.complex128),
        A_aa=np.array([[-0.4, 0.12], [0.0, -0.6 + 0.05j]], dtype=np.complex128),
    )
    full = full_matrix_from_block_operator(blocks, sector)
    recovered = block_operator_from_full(full, sector)
    assert np.allclose(recovered.A_pp, blocks.A_pp)
    assert np.allclose(recovered.A_pa, blocks.A_pa)
    assert np.allclose(recovered.A_ap, blocks.A_ap)
    assert np.allclose(recovered.A_aa, blocks.A_aa)


def test_ambient_leakage_matches_block_leakage_action() -> None:
    sector = SectorDecomposition.canonical(hidden_dim=2, observable_dim=1)
    blocks = BlockOperator(
        A_pp=np.eye(2, dtype=np.complex128),
        A_pa=np.zeros((2, 1), dtype=np.complex128),
        A_ap=np.array([[0.5, -0.25j]], dtype=np.complex128),
        A_aa=np.array([[-0.1]], dtype=np.complex128),
    )
    A = blocks.as_matrix()
    h = np.array([1.0 + 0.2j, -0.3], dtype=np.complex128)
    psi_hidden = sector.combine(h, np.zeros(1, dtype=np.complex128))
    ambient_leak = sector.observable_coordinates(leakage_full(A, sector.P, sector.Q) @ psi_hidden)
    assert np.allclose(ambient_leak, blocks.A_ap @ h)

