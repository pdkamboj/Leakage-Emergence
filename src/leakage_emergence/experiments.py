"""Reproducible theorem-facing experiments.

Every function returns a dictionary containing parameters, operators,
trajectories, metrics, and a numerical validation flag.  The examples are
finite-dimensional analogues of the theorem statements, not replacements for
the infinite-dimensional proofs.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from .dynamics import (
    duhamel_reconstruction_quad,
    first_order_prediction,
    make_linear_trajectory,
    sector_energies,
)
from .nonlinear import cubic_self_interaction, hidden_vector_field_leakage, simulate_nonlinear
from .observables import canonical_awareness_map, first_threshold_crossing, observation_norms
from .operators import BlockOperator, assemble_block_operator, full_matrix_from_block_operator
from .spaces import SectorDecomposition, projector_errors
from .theory_checks import (
    classify_eigenvectors,
    duhamel_error,
    exact_hiddenness_error,
    first_order_emergence_error,
    hidden_eigenmode_residual,
    scan_resolvent_leakage,
)
from .utils import complex_normal, estimate_loglog_slope, row_norms, seeded_rng


def exact_hiddenness_case(seed: int = 7) -> dict[str, object]:
    """Case where ``A_ap = 0`` and hidden initial states remain hidden."""

    rng = seeded_rng(seed)
    sector = SectorDecomposition.canonical(hidden_dim=3, observable_dim=2)
    A_pp = np.array(
        [[-0.15 + 0.4j, 0.2, 0.0], [-0.25, -0.1 - 0.2j, 0.1], [0.0, -0.1, -0.35]],
        dtype=np.complex128,
    )
    A_pa = complex_normal((3, 2), rng, scale=0.25)
    A_ap = np.zeros((2, 3), dtype=np.complex128)
    A_aa = np.array([[-0.4, 0.15 - 0.1j], [-0.2 + 0.05j, -0.55]], dtype=np.complex128)
    A = assemble_block_operator(A_pp, A_pa, A_ap, A_aa)
    psi0_p = np.array([1.0, -0.4 + 0.2j, 0.25j], dtype=np.complex128)
    psi0 = sector.combine(psi0_p, np.zeros(sector.observable_dim, dtype=np.complex128))
    t = np.linspace(0.0, 5.0, 151)
    trajectory = make_linear_trajectory(A, psi0, t, sector)
    max_observable_norm = exact_hiddenness_error(trajectory.psi_a)
    return {
        "name": "exact_hiddenness",
        "parameters": {"seed": seed, "hidden_dim": 3, "observable_dim": 2},
        "sector": sector,
        "blocks": BlockOperator(A_pp, A_pa, A_ap, A_aa),
        "operators": {"A": A, "P": sector.P, "Q": sector.Q},
        "trajectory": trajectory,
        "metrics": {
            "max_observable_norm": max_observable_norm,
            "projector_errors": projector_errors(sector.P, sector.Q),
        },
        "validation": bool(max_observable_norm < 1e-11),
    }


def first_order_emergence_case() -> dict[str, object]:
    """Case where ``A_ap psi0 != 0`` and observable amplitude turns on at order ``t``."""

    sector = SectorDecomposition.canonical(hidden_dim=2, observable_dim=1)
    A_pp = np.array([[-0.15 + 0.25j, 0.35], [-0.2, -0.05 - 0.1j]], dtype=np.complex128)
    A_pa = np.zeros((2, 1), dtype=np.complex128)
    A_ap = np.array([[0.9 - 0.1j, -0.25 + 0.2j]], dtype=np.complex128)
    A_aa = np.array([[-0.4 + 0.15j]], dtype=np.complex128)
    A = assemble_block_operator(A_pp, A_pa, A_ap, A_aa)
    psi0_p = np.array([1.0 + 0.2j, 0.35 - 0.15j], dtype=np.complex128)
    psi0 = sector.combine(psi0_p, np.zeros(1, dtype=np.complex128))
    t = np.linspace(0.0, 0.25, 101)
    trajectory = make_linear_trajectory(A, psi0, t, sector)
    leakage_vector = A_ap @ psi0_p
    derivative_error = first_order_emergence_error(trajectory.psi_a_coords, t, leakage_vector, samples=6)
    psi_a_norm = row_norms(trajectory.psi_a_coords)
    E_p, E_a = sector_energies(trajectory.psi_p_coords, trajectory.psi_a_coords)
    prediction = first_order_prediction(A_ap, psi0_p, t)
    return {
        "name": "first_order_emergence",
        "parameters": {"hidden_dim": 2, "observable_dim": 1},
        "sector": sector,
        "blocks": BlockOperator(A_pp, A_pa, A_ap, A_aa),
        "operators": {"A": A, "P": sector.P, "Q": sector.Q},
        "trajectory": trajectory,
        "derived": {"first_order_prediction": prediction, "E_p": E_p, "E_a": E_a},
        "metrics": {
            "leakage_norm": float(np.linalg.norm(leakage_vector)),
            "early_derivative_error": derivative_error,
            "min_positive_observable_norm_first_10": float(np.min(psi_a_norm[1:11])),
        },
        "validation": bool(np.linalg.norm(leakage_vector) > 1e-12 and np.all(psi_a_norm[1:11] > 0.0)),
    }


def duhamel_identity_case() -> dict[str, object]:
    """Compare direct observable dynamics with the Duhamel integral."""

    sector = SectorDecomposition.canonical(hidden_dim=2, observable_dim=2)
    A_pp = np.array([[-0.2 + 0.1j, 0.25], [-0.3, -0.15 - 0.05j]], dtype=np.complex128)
    A_pa = np.array([[0.05, -0.03j], [0.02 + 0.01j, -0.04]], dtype=np.complex128)
    A_ap = np.array([[0.4 - 0.1j, 0.1], [-0.2j, 0.35 + 0.05j]], dtype=np.complex128)
    A_aa = np.array([[-0.45, 0.08 - 0.04j], [-0.12 + 0.02j, -0.5 + 0.1j]], dtype=np.complex128)
    A = assemble_block_operator(A_pp, A_pa, A_ap, A_aa)
    psi0_p = np.array([0.8 - 0.2j, -0.3 + 0.4j], dtype=np.complex128)
    psi0 = sector.combine(psi0_p, np.zeros(2, dtype=np.complex128))
    t = np.linspace(0.0, 2.0, 61)
    trajectory = make_linear_trajectory(A, psi0, t, sector)

    def psi_p_at(s: float) -> np.ndarray:
        return (expm(s * A) @ psi0)[: sector.hidden_dim]

    reconstructed = duhamel_reconstruction_quad(A_aa, A_ap, psi_p_at, t)
    errors = duhamel_error(trajectory.psi_a_coords, reconstructed)
    return {
        "name": "duhamel_identity",
        "parameters": {"hidden_dim": 2, "observable_dim": 2},
        "sector": sector,
        "blocks": BlockOperator(A_pp, A_pa, A_ap, A_aa),
        "operators": {"A": A, "P": sector.P, "Q": sector.Q},
        "trajectory": trajectory,
        "derived": {"duhamel_reconstruction": reconstructed},
        "metrics": {"max_abs_error": errors.max_abs, "max_relative_error": errors.max_rel},
        "validation": bool(errors.max_abs < 1e-9),
    }


def threshold_case() -> dict[str, object]:
    """Nilpotent leakage example with an exactly known norm threshold time."""

    sector = SectorDecomposition.canonical(hidden_dim=1, observable_dim=1)
    A_pp = np.array([[0.0]], dtype=np.complex128)
    A_pa = np.array([[0.0]], dtype=np.complex128)
    A_ap = np.array([[1.0]], dtype=np.complex128)
    A_aa = np.array([[0.0]], dtype=np.complex128)
    A = assemble_block_operator(A_pp, A_pa, A_ap, A_aa)
    psi0 = np.array([1.0, 0.0], dtype=np.complex128)
    t = np.linspace(0.0, 1.0, 101)
    trajectory = make_linear_trajectory(A, psi0, t, sector)
    Pi = canonical_awareness_map(sector)
    y_norm = observation_norms(Pi, trajectory.psi)
    threshold = 0.35
    crossing = first_threshold_crossing(t, y_norm, threshold)
    return {
        "name": "threshold_detection",
        "parameters": {"threshold": threshold, "expected_crossing": threshold},
        "sector": sector,
        "blocks": BlockOperator(A_pp, A_pa, A_ap, A_aa),
        "operators": {"A": A, "P": sector.P, "Q": sector.Q, "Pi": Pi},
        "trajectory": trajectory,
        "derived": {"observation_norms": y_norm},
        "metrics": {"threshold_crossing": crossing, "absolute_error": abs(crossing - threshold)},
        "validation": bool(abs(crossing - threshold) < 1e-12),
    }


def hidden_eigenmode_case() -> dict[str, object]:
    """Hidden eigenvector remains hidden even when other hidden directions leak."""

    sector = SectorDecomposition.canonical(hidden_dim=2, observable_dim=1)
    A_pp = np.diag([0.2j, -0.4 + 0.1j]).astype(np.complex128)
    A_pa = np.zeros((2, 1), dtype=np.complex128)
    A_ap = np.array([[0.0, 0.8 - 0.1j]], dtype=np.complex128)
    A_aa = np.array([[-0.7]], dtype=np.complex128)
    A = assemble_block_operator(A_pp, A_pa, A_ap, A_aa)
    v = np.array([1.0, 0.0, 0.0], dtype=np.complex128)
    t = np.linspace(0.0, 6.0, 121)
    trajectory = make_linear_trajectory(A, v, t, sector)
    residuals = hidden_eigenmode_residual(A, v, sector)
    classifications = classify_eigenvectors(A, sector)
    max_observable_norm = exact_hiddenness_error(trajectory.psi_a)
    return {
        "name": "hidden_eigenmode",
        "parameters": {"hidden_dim": 2, "observable_dim": 1},
        "sector": sector,
        "blocks": BlockOperator(A_pp, A_pa, A_ap, A_aa),
        "operators": {"A": A, "P": sector.P, "Q": sector.Q},
        "trajectory": trajectory,
        "derived": {"classifications": classifications, "candidate_eigenvector": v},
        "metrics": {"max_observable_norm": max_observable_norm, **residuals},
        "validation": bool(max_observable_norm < 1e-11 and residuals["QAv_norm"] < 1e-12),
    }


def resolvent_leakage_case() -> dict[str, object]:
    """Compare resolvent leakage for invariant and leaky hidden sectors."""

    sector = SectorDecomposition.canonical(hidden_dim=2, observable_dim=1)
    A_pp = np.diag([-1.0, -1.8]).astype(np.complex128)
    A_pa = np.zeros((2, 1), dtype=np.complex128)
    A_aa = np.array([[-2.6]], dtype=np.complex128)
    A_ap_zero = np.zeros((1, 2), dtype=np.complex128)
    A_ap_leaky = np.array([[0.55, -0.2 + 0.1j]], dtype=np.complex128)
    A_zero = assemble_block_operator(A_pp, A_pa, A_ap_zero, A_aa)
    A_leaky = assemble_block_operator(A_pp, A_pa, A_ap_leaky, A_aa)
    lambda_values = np.linspace(0.15, 5.0, 160) + 0.35j
    zero_norms = scan_resolvent_leakage(A_zero, sector.P, sector.Q, lambda_values)
    leaky_norms = scan_resolvent_leakage(A_leaky, sector.P, sector.Q, lambda_values)
    return {
        "name": "resolvent_leakage",
        "parameters": {"lambda_imag": 0.35},
        "sector": sector,
        "operators": {"A_zero": A_zero, "A_leaky": A_leaky, "P": sector.P, "Q": sector.Q},
        "derived": {"lambda_values": lambda_values, "zero_norms": zero_norms, "leaky_norms": leaky_norms},
        "metrics": {
            "max_zero_resolvent_leakage": float(np.nanmax(zero_norms)),
            "max_leaky_resolvent_leakage": float(np.nanmax(leaky_norms)),
        },
        "validation": bool(np.nanmax(zero_norms) < 1e-12 and np.nanmax(leaky_norms) > 1e-3),
    }


def nonlinear_emergence_case() -> dict[str, object]:
    """Two cubic nonlinear examples: invariant coordinate sector and rotated leakage."""

    canonical_sector = SectorDecomposition.canonical(hidden_dim=2, observable_dim=1)
    A_preserving_blocks = BlockOperator(
        A_pp=np.diag([-0.2, -0.35]).astype(np.complex128),
        A_pa=np.zeros((2, 1), dtype=np.complex128),
        A_ap=np.zeros((1, 2), dtype=np.complex128),
        A_aa=np.array([[-0.5]], dtype=np.complex128),
    )
    A_preserving = A_preserving_blocks.as_matrix()
    N_preserving = cubic_self_interaction(lambda_nl=-0.15)
    psi0_preserving = canonical_sector.combine(np.array([0.9 + 0.1j, -0.45], dtype=np.complex128), np.zeros(1))
    t_preserving = np.linspace(0.0, 2.0, 121)
    preserving_traj = simulate_nonlinear(A_preserving, N_preserving, psi0_preserving, t_preserving, canonical_sector)
    preserving_max_observable = exact_hiddenness_error(preserving_traj.psi_a)

    hidden_basis = np.array([[1.0], [2.0], [0.35]], dtype=np.complex128)
    rotated_sector = SectorDecomposition.from_hidden_basis(hidden_basis)
    A_rotated_blocks = BlockOperator(
        A_pp=np.array([[-0.05]], dtype=np.complex128),
        A_pa=np.zeros((1, 2), dtype=np.complex128),
        A_ap=np.zeros((2, 1), dtype=np.complex128),
        A_aa=np.diag([-0.1, -0.2]).astype(np.complex128),
    )
    A_rotated = full_matrix_from_block_operator(A_rotated_blocks, rotated_sector)
    N_rotated = cubic_self_interaction(lambda_nl=0.25)
    psi0_rotated = rotated_sector.combine(np.array([0.85], dtype=np.complex128), np.zeros(2))
    t_rotated = np.linspace(0.0, 0.8, 121)
    rotated_traj = simulate_nonlinear(A_rotated, N_rotated, psi0_rotated, t_rotated, rotated_sector)
    rotated_initial_leak = hidden_vector_field_leakage(A_rotated, N_rotated, psi0_rotated, rotated_sector)
    linear_leak_norm = float(np.linalg.norm(rotated_sector.Q @ A_rotated @ rotated_sector.P))
    rotated_max_observable = exact_hiddenness_error(rotated_traj.psi_a)
    return {
        "name": "nonlinear_emergence",
        "parameters": {"preserving_lambda": -0.15, "rotated_lambda": 0.25},
        "sector": {"preserving": canonical_sector, "rotated": rotated_sector},
        "operators": {
            "A_preserving": A_preserving,
            "A_rotated": A_rotated,
            "P_rotated": rotated_sector.P,
            "Q_rotated": rotated_sector.Q,
        },
        "trajectory": {"preserving": preserving_traj, "rotated": rotated_traj},
        "metrics": {
            "preserving_max_observable_norm": preserving_max_observable,
            "rotated_initial_vector_field_leakage": rotated_initial_leak,
            "rotated_linear_leakage_norm": linear_leak_norm,
            "rotated_max_observable_norm": rotated_max_observable,
        },
        "validation": bool(
            preserving_max_observable < 1e-10
            and linear_leak_norm < 1e-12
            and rotated_initial_leak > 1e-4
            and rotated_max_observable > 1e-4
        ),
    }


def early_time_scaling_case() -> dict[str, object]:
    """Show ``||psi_a(t)|| = O(t)`` while ``E_a(t) = O(t^2)``."""

    sector = SectorDecomposition.canonical(hidden_dim=1, observable_dim=1)
    A = assemble_block_operator(
        np.array([[0.0]], dtype=np.complex128),
        np.array([[0.0]], dtype=np.complex128),
        np.array([[1.25]], dtype=np.complex128),
        np.array([[0.0]], dtype=np.complex128),
    )
    psi0 = np.array([1.0, 0.0], dtype=np.complex128)
    t = np.concatenate([[0.0], np.logspace(-5, -1, 90)])
    trajectory = make_linear_trajectory(A, psi0, t, sector)
    psi_a_norm = row_norms(trajectory.psi_a_coords)
    _E_p, E_a = sector_energies(trajectory.psi_p_coords, trajectory.psi_a_coords)
    amplitude_slope = estimate_loglog_slope(t[1:], psi_a_norm[1:])
    energy_slope = estimate_loglog_slope(t[1:], E_a[1:])
    return {
        "name": "early_time_scaling",
        "parameters": {"leakage_strength": 1.25},
        "sector": sector,
        "operators": {"A": A, "P": sector.P, "Q": sector.Q},
        "trajectory": trajectory,
        "derived": {"observable_norm": psi_a_norm, "E_a": E_a},
        "metrics": {"amplitude_loglog_slope": amplitude_slope, "energy_loglog_slope": energy_slope},
        "validation": bool(abs(amplitude_slope - 1.0) < 1e-10 and abs(energy_slope - 2.0) < 1e-10),
    }


def run_all_experiments() -> dict[str, dict[str, object]]:
    """Run the complete reproducible experiment suite."""

    experiments = [
        exact_hiddenness_case(),
        first_order_emergence_case(),
        duhamel_identity_case(),
        threshold_case(),
        hidden_eigenmode_case(),
        resolvent_leakage_case(),
        nonlinear_emergence_case(),
        early_time_scaling_case(),
    ]
    return {str(result["name"]): result for result in experiments}

