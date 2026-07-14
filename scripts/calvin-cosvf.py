"""
Limited foresight run — solves the annual COSVF sequence with fixed penalty parameters.

Model dir: my-models/calvin-cosvf/
  links.csv                — single water-year template network
  cosvf-params.csv         — penalty parameters (from EA or manual)
  r-dict.json              — reservoir metadata
  inflows.csv              — monthly inflows for full period
  variable-constraints.csv — time-varying link bounds
  results/                 — output CSVs written here

Input files are prepared automatically if absent.  To regenerate them manually:
  delete my-models/calvin-cosvf/links.csv and re-run this script.

Configuration is read from my-models/calvin-cosvf/config.toml (auto-created from
scripts/configs/calvin-cosvf.toml on first run).  Edit the model-dir copy to
customise a specific experiment without touching the committed defaults.
"""
import os
import sys
from calvin import cosvfea

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report import generate_report
from run_config import load_config

MODEL_DIR  = './my-models/calvin-cosvf'
cfg        = load_config('calvin-cosvf', MODEL_DIR)

DATA_PATH  = cfg['run']['data_path']
SOLVER     = cfg['run']['solver']
NPROC      = cfg['run']['nproc']
_net       = cfg['network']
NODE_LB_OVERRIDES        = _net.get('node_lb_overrides', {})
STORAGE_PERSUASION_COST  = _net.get('storage_persuasion_cost', -0.02)
CONNECTOR_COST           = _net.get('connector_cost', 0.01)

LINKS_FILE = os.path.join(MODEL_DIR, 'links.csv')
RESULT_DIR = os.path.join(MODEL_DIR, 'results')

os.makedirs(RESULT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Prepare input files if not already present
# ---------------------------------------------------------------------------
if not os.path.isfile(LINKS_FILE):
    print('Preparing COSVF input files...')
    from calvin.network.prepare import prepare_cosvf
    prepare_cosvf(DATA_PATH, MODEL_DIR,
                  node_lb_overrides=NODE_LB_OVERRIDES,
                  storage_persuasion_cost=STORAGE_PERSUASION_COST,
                  connector_cost=CONNECTOR_COST)
    print(f'  Input files written to {MODEL_DIR}')

# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------
calvin = cosvfea.COSVF(pwd=MODEL_DIR)
calvin.create_pyomo_model(debug_mode=True)

f1, f2, f3 = calvin.cosvf_solve(solver=SOLVER, nproc=NPROC, resultdir=RESULT_DIR,
                                 show_progress=True)
print('total cost={} $M/yr'.format(round(f1, 2)))
print('gw overdraft={} MAF/yr'.format(round(f2, 2)))
generate_report(MODEL_DIR)
