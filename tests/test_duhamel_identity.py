from __future__ import annotations

from leakage_emergence.experiments import duhamel_identity_case


def test_duhamel_identity_consistency() -> None:
    result = duhamel_identity_case()
    assert result["validation"]
    assert result["metrics"]["max_abs_error"] < 1e-9
    assert result["metrics"]["max_relative_error"] < 1e-7

