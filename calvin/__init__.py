__version__ = "2026.02.10"

try:
    from .calvin import CALVIN
    from .capacity import CALVINCap
    from .postprocessor import postprocess, aggregate_regions, combine_annual_results
    from .extensive_form import build_ef, solve_ef
except ImportError:
    pass  # pyomo not available (e.g. when running the web app only)

from .network import load_network, build_matrix, export_matrix
from .ensemble import draw_samples, mean_sample, wa_wi_cloud  # df-pure sampler