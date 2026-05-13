from __future__ import annotations

import numpy as np

from leakage_emergence.experiments import exact_hiddenness_case


def test_exact_hiddenness_when_A_ap_zero() -> None:
    result = exact_hiddenness_case()
    trajectory = result["trajectory"]
    assert result["validation"]
    assert result["metrics"]["max_observable_norm"] < 1e-11
    assert np.max(np.linalg.norm(trajectory.psi_a_coords, axis=1)) < 1e-11

