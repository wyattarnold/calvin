"""
Perfect foresight run — solves the full 82-year network as a single LP.

Model dir: my-models/calvin-pf/ by default, or pass one as the first
argument (verbatim path, e.g. ./my-models/calvin-pf-scaled):
  links82yr.csv  — time-expanded network (auto-generated on first run)
  results/       — output CSVs written here

The links file is built automatically if absent.  To regenerate it manually:
  delete my-models/calvin-pf/links82yr.csv and re-run this script.

Configuration is read from my-models/calvin-pf/config.toml (auto-created from
scripts/configs/calvin-pf.toml on first run).  Edit the model-dir copy to
customise a specific experiment without touching the committed defaults.
"""
import os
import sys
import numpy as np
from calvin import CALVIN, postprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report import generate_report
from run_config import load_config

MODEL_DIR  = sys.argv[1] if len(sys.argv) > 1 else './my-models/calvin-pf'
cfg        = load_config('calvin-pf', MODEL_DIR)

DATA_PATH  = cfg['run']['data_path']
SOLVER     = cfg['run']['solver']
NPROC      = cfg['run']['nproc']
NODE_LB_OVERRIDES = cfg['network'].get('node_lb_overrides', {})

LINKS_FILE = os.path.join(MODEL_DIR, 'links82yr.csv')
RESULT_DIR = os.path.join(MODEL_DIR, 'results')

os.makedirs(RESULT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# HiGHS basis warm start (mirrors calvin-pf-cap.py): every solve saves its
# optimal basis to results/highs.bas; a variant run with the same links file
# and only bound changes (scenario edits) can pre-load a baseline's basis
# via [run] warm_start_basis. NOTE: plain-PF bases and
# pf-cap bases are not interchangeable — the capacity model adds X_cap
# columns and coupling rows, so the LP shapes differ.
# ---------------------------------------------------------------------------
SOLVER_OPTIONS = dict(cfg['run'].get('solver_options', {}))
if SOLVER == 'highs':
    SOLVER_OPTIONS.setdefault('write_basis_file',
                              os.path.join(RESULT_DIR, 'highs.bas'))
    warm = cfg['run'].get('warm_start_basis', '')
    if warm:
        if not os.path.isabs(warm):
            warm = os.path.normpath(os.path.join(MODEL_DIR, warm))
        if os.path.isfile(warm):
            SOLVER_OPTIONS['read_basis_file'] = warm
            if SOLVER_OPTIONS.pop('solver', None):
                print('warm start: dropping solver algorithm override so '
                      'the basis is used')
            print('warm start from basis: %s' % warm)
        else:
            print('warm_start_basis not found, starting cold: %s' % warm)

# ---------------------------------------------------------------------------
# Build links file if not already present
# ---------------------------------------------------------------------------
if not os.path.isfile(LINKS_FILE):
    print('Building 82-year links file...')
    from calvin.network.loader import load_network
    from calvin.network.matrix import build_matrix

    network = load_network(DATA_PATH)
    df = build_matrix(network, start='1921-10', stop='2003-09',
                      add_debug=False, node_lb_overrides=NODE_LB_OVERRIDES)
    df.to_csv(LINKS_FILE, index=False)
    print(f'  Wrote {len(df)} links to {LINKS_FILE}')

# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------
calvin = CALVIN(linksfile=LINKS_FILE, logdir=MODEL_DIR)
calvin.create_pyomo_model(debug_mode=False)
try:
    solved = calvin.solve_pyomo_model(solver=SOLVER, nproc=NPROC,
                                      debug_mode=False,
                                      solver_options=SOLVER_OPTIONS)
except ValueError:
    # stale/mismatched basis file: HiGHS rejects it and the appsi result
    # parser chokes. Retry cold.
    if 'read_basis_file' not in SOLVER_OPTIONS:
        raise
    print('warm start failed (basis does not match this model); '
          'retrying cold')
    SOLVER_OPTIONS.pop('read_basis_file')
    solved = calvin.solve_pyomo_model(solver=SOLVER, nproc=NPROC,
                                      debug_mode=False,
                                      solver_options=SOLVER_OPTIONS)
if not solved:
    sys.exit('Solve did not reach an optimal solution; see log. '
             'No results written.')

model = calvin.model_to_dataframe()

def compute_network_costs(model_df):
    cost_links = model_df.drop(model_df[((model_df['i'].str.contains('SR')) |
                                        (model_df['i'].str.contains('GW'))) &
                                        (model_df['j'].str.contains('FINAL'))].index)
    cost_links = cost_links.loc[~cost_links.index.str.contains('DBUG')]
    cost_links = cost_links.drop(cost_links[(cost_links['i'].str.contains('SR')) &
                                            (cost_links['j'].str.contains('SR'))].index)
    short_links = cost_links.loc[cost_links['cost'] < 0]
    short_links = short_links.loc[short_links.upper_bound < 1e6]
    short_costs = -1 * ((short_links.upper_bound - short_links.flow) * short_links.cost).sum()
    op_links = cost_links.loc[cost_links['cost'] > 0]
    op_costs = (op_links.flow * op_links.cost).sum()
    return short_costs, op_costs

short_cost, op_cost = compute_network_costs(model)
print('total cost={} $M/yr'.format(round((short_cost + op_cost) / 82 / 1e3, 2)))
print('short cost={} $M/yr'.format(round(short_cost / 82 / 1e3, 2)))
print('op cost={} $M/yr'.format(round(op_cost / 82 / 1e3, 2)))

gw_initial = model.loc[(model.index.str.contains('GW_')) & (model.index.str.contains('INITIAL'))]
gw_final   = model.loc[(model.index.str.contains('GW_')) & (model.index.str.contains('FINAL'))]
gw_change  = gw_final['flow'].values - gw_initial['lower_bound'].values
gw_od      = gw_change[np.where(gw_change < 0)].sum()
print('gw overdraft={} MAF/yr'.format(round(gw_od / 82 / 1e3, 2)))

postprocess(calvin.df, calvin.model, RESULT_DIR)
generate_report(MODEL_DIR)
