#!/usr/bin/env python3
"""Run the reproducible linear theorem examples."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leakage_emergence import experiments  # noqa: E402


def main() -> None:
    names = [
        "exact_hiddenness_case",
        "first_order_emergence_case",
        "duhamel_identity_case",
        "threshold_case",
        "hidden_eigenmode_case",
        "resolvent_leakage_case",
        "early_time_scaling_case",
    ]
    for name in names:
        result = getattr(experiments, name)()
        print(f"\n{name}: validation={result['validation']}")
        for key, value in result["metrics"].items():
            if isinstance(value, dict):
                compact = {k: f"{v:.3e}" for k, v in value.items()}
                print(f"  {key}: {compact}")
            elif isinstance(value, float):
                print(f"  {key}: {value:.6e}")
            else:
                print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

