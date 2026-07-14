"""
calvin.network — Python replacement for calvin-network-tools.

Reads the calvin-network-data repository and produces the time-expanded
network matrix (i, j, k, cost, amplitude, lower_bound, upper_bound) consumed
by the CALVIN Pyomo solver.
"""

from .loader import load_network
from .matrix import build_matrix, build_annual_matrix, export_matrix
from .prepare import prepare_cosvf, prepare_pf_astep, prepare_cosvf_astep

__all__ = [
    "load_network", "build_matrix", "build_annual_matrix", "export_matrix",
    "prepare_cosvf", "prepare_pf_astep", "prepare_cosvf_astep",
]
