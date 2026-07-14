"""
Perfect foresight run — solves the full 82-year network as a single LP.

Model dir: my-models/calvin-pf/
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

MODEL_DIR  = './my-models/calvin-pf'
cfg        = load_config('calvin-pf', MODEL_DIR)

DATA_PATH  = cfg['run']['data_path']
SOLVER     = cfg['run']['solver']
NPROC      = cfg['run']['nproc']
NODE_LB_OVERRIDES = cfg['network'].get('node_lb_overrides', {})

LINKS_FILE = os.path.join(MODEL_DIR, 'links82yr.csv')
RESULT_DIR = os.path.join(MODEL_DIR, 'results')

os.makedirs(RESULT_DIR, exist_ok=True)

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
calvin.solve_pyomo_model(solver=SOLVER, nproc=NPROC, debug_mode=False)

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
