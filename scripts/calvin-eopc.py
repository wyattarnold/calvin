"""
Annual constraint-based run — solves one water year at a time using storage fraction constraints.

Model dir: my-models/calvin-eopc/
  links/linksWY{year}.csv  — per-year networks (auto-generated on first run)
  config.toml              — run configuration (copied from scripts/configs/ on first run)
  results/annual/WY{year}/ — per-year output CSVs
  results/                 — combined output CSVs (written after loop)

Per-year links files are built automatically if the links/ directory is absent.
To regenerate them manually: delete my-models/calvin-eopc/links/ and re-run.

Configuration is loaded from my-models/calvin-eopc/config.toml (created
automatically from scripts/configs/calvin-eopc.toml on first run).
"""
import os
import sys
import pandas as pd
from calvin import CALVIN, postprocess, combine_annual_results

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report import generate_report
from run_config import load_config

MODEL_DIR = './my-models/calvin-eopc'
cfg = load_config('calvin-eopc', MODEL_DIR)

DATA_PATH         = cfg['run']['data_path']
SOLVER            = cfg['run'].get('solver', 'highs')
EOP_FRACTION      = cfg['run'].get('eop_fraction', 0.1)
NODE_LB_OVERRIDES = cfg.get('network', {}).get('node_lb_overrides', {})

LINKS_DIR  = os.path.join(MODEL_DIR, 'links')
ANNUAL_DIR = os.path.join(MODEL_DIR, 'results', 'annual')
RESULT_DIR = os.path.join(MODEL_DIR, 'results')
os.makedirs(LINKS_DIR, exist_ok=True)
os.makedirs(ANNUAL_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Build per-year links files if not already present
# ---------------------------------------------------------------------------
if not os.path.isfile(os.path.join(LINKS_DIR, 'linksWY1922.csv')):
    print('Building per-year links files (WY1922–2003)...')
    from calvin.network.loader import load_network
    from calvin.network.matrix import build_matrix

    network = load_network(DATA_PATH)
    for wy in range(1922, 2004):
        fp = os.path.join(LINKS_DIR, f'linksWY{wy}.csv')
        df = build_matrix(network, start=f'{wy-1}-10', stop=f'{wy}-09',
                          add_debug=True, node_lb_overrides=NODE_LB_OVERRIDES)
        df.to_csv(fp, index=False)
    print(f'  Per-year links written to {LINKS_DIR}')

# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------
eop = None
all_adjustments = []

for wy in range(1922, 2004):
    print(f'\nNow running WY {wy}')

    wy_dir = os.path.join(ANNUAL_DIR, f'WY{wy}')
    calvin = CALVIN(os.path.join(LINKS_DIR, f'linksWY{wy}.csv'), ic=eop,
                    log_name=f'WY{wy}', logdir=wy_dir)
    calvin.eop_constraint_multiplier(EOP_FRACTION)

    calvin.create_pyomo_model(debug_mode=True, debug_cost=2e8)
    debug_converged = calvin.solve_pyomo_model(solver=SOLVER, nproc=1, debug_mode=True, maxiter=15)

    if debug_converged:
        calvin.create_pyomo_model(debug_mode=False)
        calvin.solve_pyomo_model(solver=SOLVER, nproc=1, debug_mode=False)
    else:
        print(f'WY {wy}: debug did not converge; using last debug solution for postprocessing')

    adj = calvin.get_bound_adjustments()
    if not adj.empty:
        adj.insert(0, 'water_year', wy)
        all_adjustments.append(adj)

    eop = postprocess(calvin.df, calvin.model, resultdir=wy_dir, annual=True)

# ---------------------------------------------------------------------------
# Combine and save
# ---------------------------------------------------------------------------
print('\nCombining annual results...')
combine_annual_results(range(1922, 2004), annual_dir=ANNUAL_DIR, output_dir=RESULT_DIR)
print(f'Combined results written to {RESULT_DIR}')

if all_adjustments:
    adj_df = pd.concat(all_adjustments, ignore_index=True)
    adj_path = os.path.join(RESULT_DIR, 'bound_adjustments.csv')
    adj_df.to_csv(adj_path, index=False)
    print(f'Bound adjustments written to {adj_path} ({len(adj_df)} rows)')

generate_report(MODEL_DIR)
