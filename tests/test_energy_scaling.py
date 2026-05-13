from __future__ import annotations

from leakage_emergence.experiments import early_time_scaling_case


def test_observable_amplitude_and_energy_scaling() -> None:
    result = early_time_scaling_case()
    assert result["validation"]
    assert abs(result["metrics"]["amplitude_loglog_slope"] - 1.0) < 1e-10
    assert abs(result["metrics"]["energy_loglog_slope"] - 2.0) < 1e-10

