from __future__ import annotations

import numpy as np

from leakage_emergence.experiments import threshold_case
from leakage_emergence.observables import first_threshold_crossing


def test_threshold_detection_correctness() -> None:
    result = threshold_case()
    assert result["validation"]
    assert abs(result["metrics"]["threshold_crossing"] - 0.35) < 1e-12


def test_threshold_detection_interpolates_between_samples() -> None:
    t = np.array([0.0, 1.0, 2.0])
    values = np.array([0.0, 0.2, 0.6])
    assert abs(first_threshold_crossing(t, values, 0.4) - 1.5) < 1e-12
    assert np.isinf(first_threshold_crossing(t, values, 2.0))

