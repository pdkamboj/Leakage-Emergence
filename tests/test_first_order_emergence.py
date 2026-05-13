from __future__ import annotations

import numpy as np

from leakage_emergence.experiments import first_order_emergence_case


def test_first_order_emergence_matches_initial_derivative() -> None:
    result = first_order_emergence_case()
    blocks = result["blocks"]
    trajectory = result["trajectory"]
    psi0_p = trajectory.psi_p_coords[0]
    leakage = blocks.A_ap @ psi0_p
    assert result["validation"]
    assert np.linalg.norm(leakage) > 1e-12
    assert result["metrics"]["early_derivative_error"] < 5e-3
    assert np.all(np.linalg.norm(trajectory.psi_a_coords[1:10], axis=1) > 0.0)

