from __future__ import annotations

from leakage_emergence.experiments import nonlinear_emergence_case


def test_nonlinear_hiddenness_and_leakage_examples() -> None:
    result = nonlinear_emergence_case()
    metrics = result["metrics"]
    assert result["validation"]
    assert metrics["preserving_max_observable_norm"] < 1e-10
    assert metrics["rotated_linear_leakage_norm"] < 1e-12
    assert metrics["rotated_initial_vector_field_leakage"] > 1e-4
    assert metrics["rotated_max_observable_norm"] > 1e-4

