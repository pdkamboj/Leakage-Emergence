from __future__ import annotations

import numpy as np

from leakage_emergence.observables import general_awareness_map
from leakage_emergence.spaces import SectorDecomposition, projector_errors, verify_projectors


def test_canonical_projector_identities() -> None:
    sector = SectorDecomposition.canonical(hidden_dim=3, observable_dim=2)
    assert verify_projectors(sector.P, sector.Q, tol=1e-13)
    errors = projector_errors(sector.P, sector.Q)
    assert max(errors.values()) < 1e-13
    assert np.allclose(sector.P @ sector.Q, 0.0)
    assert np.allclose(sector.P + sector.Q, np.eye(5))


def test_general_projector_and_awareness_kernel() -> None:
    hidden_basis = np.array([[1.0, 0.0], [1.0j, 1.0], [0.0, 2.0], [0.5, -0.3j]], dtype=np.complex128)
    sector = SectorDecomposition.from_hidden_basis(hidden_basis)
    Pi = general_awareness_map(sector, output_dim=3, seed=11)
    assert verify_projectors(sector.P, sector.Q, tol=1e-12)
    assert np.linalg.norm(Pi @ sector.hidden_basis) < 1e-12
    assert np.linalg.matrix_rank(Pi @ sector.observable_basis) == sector.observable_dim

