from __future__ import annotations

from leakage_emergence.experiments import hidden_eigenmode_case


def test_hidden_eigenmode_remains_hidden() -> None:
    result = hidden_eigenmode_case()
    assert result["validation"]
    assert result["metrics"]["QAv_norm"] < 1e-12
    assert result["metrics"]["max_observable_norm"] < 1e-11
    assert any(record["kind"] == "hidden" for record in result["derived"]["classifications"])

