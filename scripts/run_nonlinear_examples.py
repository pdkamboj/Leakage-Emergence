#!/usr/bin/env python3
"""Run the reproducible nonlinear examples."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leakage_emergence.experiments import nonlinear_emergence_case  # noqa: E402


def main() -> None:
    result = nonlinear_emergence_case()
    print(f"nonlinear_emergence_case: validation={result['validation']}")
    for key, value in result["metrics"].items():
        print(f"  {key}: {value:.6e}")


if __name__ == "__main__":
    main()

