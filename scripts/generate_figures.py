#!/usr/bin/env python3
"""Generate all supplement figures."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leakage_emergence.plotting import generate_all_figures  # noqa: E402


def main() -> None:
    paths = generate_all_figures(ROOT / "figures")
    print("Generated figure files:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
