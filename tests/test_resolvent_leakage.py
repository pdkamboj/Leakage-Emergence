from __future__ import annotations

import numpy as np

from leakage_emergence.experiments import resolvent_leakage_case
from leakage_emergence.theory_checks import resolvent_leakage_norm


def test_resolvent_leakage_zero_vs_nonzero() -> None:
    result = resolvent_leakage_case()
    assert result["validation"]
    assert result["metrics"]["max_zero_resolvent_leakage"] < 1e-12
    assert result["metrics"]["max_leaky_resolvent_leakage"] > 1e-3

    sector = result["sector"]
    lam = 1.25 + 0.35j
    zero_norm = resolvent_leakage_norm(result["operators"]["A_zero"], sector.P, sector.Q, lam)
    leaky_norm = resolvent_leakage_norm(result["operators"]["A_leaky"], sector.P, sector.Q, lam)
    assert np.isclose(zero_norm, 0.0, atol=1e-12)
    assert leaky_norm > 1e-3

