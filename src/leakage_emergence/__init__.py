"""Finite-dimensional leakage-induced emergence simulations.

The package implements a mathematically honest finite-dimensional analogue of
the Hilbert-space model in the paper.  It exposes projector geometry, block
operators, exact linear dynamics, nonlinear ODE dynamics, observation maps,
theorem diagnostics, reproducible experiments, and plotting utilities.
"""

from .dynamics import LinearTrajectory, exact_linear_flow, make_linear_trajectory, sector_energies
from .nonlinear import NonlinearTrajectory, cubic_self_interaction, sector_preserving_cubic, simulate_nonlinear
from .observables import canonical_awareness_map, first_threshold_crossing, general_awareness_map, observe
from .operators import BlockOperator, assemble_block_operator, split_block_operator
from .spaces import SectorDecomposition, projector_errors, projector_from_basis, verify_projectors
from .theory_checks import resolvent_leakage, resolvent_leakage_norm

__all__ = [
    "BlockOperator",
    "LinearTrajectory",
    "NonlinearTrajectory",
    "SectorDecomposition",
    "assemble_block_operator",
    "canonical_awareness_map",
    "cubic_self_interaction",
    "exact_linear_flow",
    "first_threshold_crossing",
    "general_awareness_map",
    "make_linear_trajectory",
    "observe",
    "projector_errors",
    "projector_from_basis",
    "resolvent_leakage",
    "resolvent_leakage_norm",
    "sector_energies",
    "sector_preserving_cubic",
    "simulate_nonlinear",
    "split_block_operator",
    "verify_projectors",
]

