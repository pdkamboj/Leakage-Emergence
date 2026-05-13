# Leakage-Emergence
Finite-dimensional numerical supplement and reproducible Python codebase for the mathematical physics paper "Leakage-Induced Emergence in Projected Linear and Nonlinear Cognitive Fields.


# Leakage-Induced Emergence: Numerical Supplement

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
This repository contains the finite-dimensional numerical supplement for the theorem paper **"Leakage-Induced Emergence in Projected Linear and Nonlinear Cognitive Fields."** The provided Python package implements reproducible, projected linear and nonlinear dynamics to illustrate the mechanisms of emergence in finite-dimensional complex vector spaces. It is designed to generate paper-ready figures and provide transparent numerical diagnostics that support the infinite-dimensional operator-theoretic proofs presented in the main manuscript.

## Theoretical Context

The accompanying mathematical framework studies an emergence problem for projected dynamical systems on Hilbert space. Given a complex state space and a bounded observation map $\Pi$, the system decomposes into a hidden ("preconscious") sector $\mathcal{P} = \ker(\Pi)$ and an orthogonal observable sector $\mathcal{W} = \mathcal{P}^\perp$. 

The central structural object of this theory is the **leakage operator**, $A_{ap} := QAP$, which specifically measures how hidden states are transported into the observable sector over time.

## What the Code Illustrates

This software package provides controlled, finite-dimensional ($\mathbb{C}^n$) analogues of the paper's core theorems, including:
* **Exact Hiddenness:** Demonstrates that if $A_{ap} = 0$, trajectories starting in the hidden sector remain completely hidden.
* **First-Order Emergence:** Validates the analytical prediction that observable amplitude emerges linearly over time when a hidden initial state is actively coupled to the observable sector.
* **Duhamel Reconstruction:** Verifies that direct observable dynamics match the variation-of-constants integral formula.
* **Thresholded Observation:** Computes the exact first crossing time for a given observation norm threshold.
* **Spectral & Resolvent Leakage:** Tests hidden eigenmodes and evaluates the leakage resolvent norm.
* **Energy Onset Scaling:** Confirms the distinct early-time scaling laws where observable amplitude scales as $\mathcal{O}(t)$, while observable-sector energy scales as $\mathcal{O}(t^2)$.
* **Nonlinear Dynamics:** Integrates cubic nonlinearities to contrast hiddenness-preserving nonlinearities against leak-inducing ones.

## Installation

Ensure you have Python 3.11 or higher installed. Clone the repository and install the package and its dependencies (NumPy, SciPy, Matplotlib, Pytest) using `pip`:

```bash
git clone [https://github.com/YOUR-USERNAME/leakage-emergence.git](https://github.com/YOUR-USERNAME/leakage-emergence.git)
cd leakage-emergence
pip install -e .
