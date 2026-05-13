HEAD
# Leakage-Induced Emergence

Finite-dimensional numerical supplement for the theorem paper **"Leakage-Induced Emergence in Projected Linear and Nonlinear Cognitive Fields."**

The repository implements a reproducible Python package for projected linear and nonlinear dynamics.  Its purpose is to illustrate theorem mechanisms in finite-dimensional complex vector spaces, generate paper-ready figures, and provide transparent numerical diagnostics.  It is not a proof of the infinite-dimensional theory and it is not an empirical model of consciousness.

## Model

The theorem paper starts with a Hilbert space `H`, a bounded observation or awareness map `Pi`, and the hidden sector

```text
P_sector = ker(Pi)
W_sector = P_sector^\perp
```

This code uses the finite-dimensional analogue `H = C^n` with the standard Hermitian inner product.  Every state is decomposed by complementary orthogonal projectors:

```text
psi(t) = psi_p(t) + psi_a(t)
psi_p(t) = P psi(t)
psi_a(t) = Q psi(t)
P + Q = I,   P Q = Q P = 0
```

Relative to `C^n = P_sector direct-sum W_sector`, a linear generator is assembled as

```text
A = [[A_pp, A_pa],
     [A_ap, A_aa]]
```

The leakage block `A_ap = Q A P` maps hidden states into the observable sector.  The experiments are built around this block because the theorem paper identifies it as the structural source of hidden-to-observable emergence.

## What The Numerics Illustrate

The package contains reproducible finite-dimensional examples for:

- **Exact hiddenness**: if `A_ap = 0` and `psi(0)` is hidden, then the observable component stays zero.
- **First-order emergence**: if `A_ap psi_0 != 0`, then `psi_a(t) = t A_ap psi_0 + o(t)` at early time.
- **Duhamel reconstruction**: direct observable dynamics agree with the variation-of-constants formula.
- **Thresholded observation**: the observation norm `||Pi psi(t)||` has a computed first crossing time `t_Theta`.
- **Spectral hiddenness**: hidden eigenvectors remain hidden under the flow.
- **Resolvent leakage**: `Q (lambda I - A)^(-1) P` detects hidden-to-observable mixing.
- **Nonlinear preservation and leakage**: cubic nonlinearities can preserve a hidden coordinate sector or induce observable leakage in a rotated sector.
- **Energy onset**: observable amplitude scales like `t`, while aware-sector energy scales like `t^2`.

These are theorem-supporting numerical illustrations in finite dimension.  They do not prove the infinite-dimensional statements.

## Install

From the repository root:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[test]"
```

The package requires Python 3.11+ and uses NumPy, SciPy, Matplotlib, and Pytest.  No JAX, PyTorch, seaborn, or system LaTeX installation is required.

## Run Experiments

Linear theorem examples:

```bash
python scripts/run_linear_examples.py
```

Nonlinear examples:

```bash
python scripts/run_nonlinear_examples.py
```

The scripts print validation flags and numerical metrics such as leakage norms, Duhamel reconstruction errors, threshold times, and scaling slopes.

## Generate Figures

```bash
python scripts/generate_figures.py
```

The script is a lightweight wrapper around `leakage_emergence.plotting.generate_all_figures`.  It writes vector PDF and high-resolution PNG outputs to `figures/`, and also writes:

```text
figures/figure_captions.tex
```

The caption file contains suggested LaTeX `figure` environments, labels, captions, and theorem-support notes for paper integration.

## Core Figure Set

- `01_component_norms.pdf/png`: hidden and observable norms, illustrating leakage from hidden to observable sector.
- `04_threshold_crossing.pdf/png`: observation norm, threshold `Theta`, and first crossing time `t_Theta`.
- `08_early_time_scaling.pdf/png`: log-log observable amplitude and aware-sector energy scaling, with fitted slopes near 1 and 2.
- `07_nonlinear_emergence.pdf/png`: comparison of hiddenness-preserving and leak-inducing nonlinear examples.

Additional theorem-supporting figures are also generated:

- `02_first_order_emergence.pdf/png`: direct observable amplitude versus the first-order prediction.
- `03_duhamel_agreement.pdf/png`: direct observable dynamics versus Duhamel reconstruction.
- `05_spectral_hiddenness.pdf/png`: hidden eigenmode behavior and eigenvector sector content.
- `06_resolvent_leakage.pdf/png`: leakage-resolvent norm for invariant and leaky block operators.

## Tests

```bash
pytest -q
```

The tests cover projector identities, awareness-map kernels, exact hiddenness, first-order emergence, Duhamel consistency, threshold interpolation, spectral hiddenness, resolvent leakage, nonlinear preservation/leakage, and early-time energy scaling.

## Package Layout

```text
src/leakage_emergence/
  spaces.py        sector decompositions and orthogonal projectors
  operators.py     block operator assembly and coordinate conversion
  dynamics.py      exact matrix-exponential flow, ODE flow, Duhamel integrals
  nonlinear.py     cubic nonlinearities and nonlinear ODE integration
  observables.py   awareness maps, scalar observables, threshold detection
  theory_checks.py theorem-facing numerical diagnostics
  experiments.py   named reproducible experiment cases
  plotting.py      publication-style figure and caption generation
  utils.py         reproducibility and numerical helpers
```

## Related Context

The framework sits naturally inside invariant-subspace and projected-dynamical-systems reasoning.  Exact hiddenness is an invariance statement for `ker(Pi)`.  First-order emergence records the failure of that invariance through the block `Q A P`.  The Duhamel identity is a finite-dimensional variation-of-constants formula for the observable block equation.  The resolvent diagnostic is a frequency-domain view of the same hidden-to-observable mixing mechanism, and the threshold construction connects projected dynamics to observability-style detection.

No literature references are managed in this repository yet.  A manuscript can add citations for invariant subspaces, semigroup evolution, resolvent methods, and observability theory in the paper bibliography.

## Limitations

This package is a finite-dimensional numerical supplement.  Matrix exponentials, ODE solves, and floating-point diagnostics can illustrate theorem mechanisms in `C^n`, but they do not replace a functional-analytic proof on an infinite-dimensional Hilbert space.  Numerical tolerances, finite grids, and matrix conditioning all matter.

The cognitive language is interpretive.  The code studies operator-theoretic projected dynamics and should not be framed as proving biological consciousness.

=======
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
a336f0b797dc151d46f1e34a84d0ea3fa30db46b
