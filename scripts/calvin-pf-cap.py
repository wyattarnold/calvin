"""
Perfect-foresight capacity expansion run (Phase 1).

Solves the 82-year network as a single LP with endogenous facility capacity:
X_cap variables on the candidate facilities in calvin/data/facilities.csv,
coupled to every monthly facility flow, with annualized capital cost in the
objective. See tmp/notes/facilities-mapping.md for conventions.

Model dir: my-models/calvin-pf-cap/
  links82yr.csv  — time-expanded network (auto-generated on first run)
  results/       — standard postprocess CSVs plus capacity.csv and
                   capacity_duals.csv

Configuration is read from my-models/calvin-pf-cap/config.toml (auto-created
from scripts/configs/calvin-pf-cap.toml on first run).
"""
import os
import sys
import numpy as np
from calvin import CALVINCap, postprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report import generate_report
from run_config import load_config

MODEL_DIR  = sys.argv[1] if len(sys.argv) > 1 else './my-models/calvin-pf-cap'
cfg        = load_config('calvin-pf-cap', MODEL_DIR)

DATA_PATH  = cfg['run']['data_path']
SOLVER     = cfg['run']['solver']
NPROC      = cfg['run']['nproc']
NODE_LB_OVERRIDES = cfg['network'].get('node_lb_overrides', {})
# 'none' (default) = overdraft allowed; 'gw' = no net GW overdraft (each
# basin's 82-yr ending storage >= initial; the links file gains a GW->FINAL
# lower bound and is named links82yr-nogwod.csv so modes never clobber)
CONSTRAIN_ENDING = cfg['network'].get('constrain_ending', 'none')
CAP        = cfg['capacity']

SCENARIO = cfg.get('scenario') or None
if SCENARIO and SCENARIO.get('enabled', True):
    ref = SCENARIO.get('reference_results', '../calvin-pf-cap-market/results')
    if not os.path.isabs(ref):
        ref = os.path.normpath(os.path.join(MODEL_DIR, ref))
    if not os.path.isfile(os.path.join(ref, 'flow.csv')):
        sys.exit('[scenario] reference results not found: %s\n'
                 'Run the unconstrained market baseline first, e.g.:\n'
                 '  python scripts/calvin-pf-cap.py ./my-models/calvin-pf-cap-market\n'
                 '(with [scenario] enabled = false in its config.toml), or point\n'
                 '[scenario].reference_results at an existing unconstrained '
                 'results dir.' % ref)
    SCENARIO = {**SCENARIO, 'reference_results': ref}
else:
    SCENARIO = None

LINKS_FILE = os.path.join(
    MODEL_DIR,
    {'none': 'links82yr.csv', 'gw': 'links82yr-nogwod.csv',
     'all': 'links82yr-all.csv'}[CONSTRAIN_ENDING])
RESULT_DIR = os.path.join(MODEL_DIR, 'results')

os.makedirs(RESULT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# HiGHS basis warm start: every solve saves its optimal basis to
# results/highs.bas; a variant run (same links + facilities, only bounds
# changed) can start from a baseline's basis via [run] warm_start_basis.
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
            # an algorithm override (e.g. ipm) would ignore the basis;
            # let HiGHS pick simplex from the loaded basis instead
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
                      add_debug=False, node_lb_overrides=NODE_LB_OVERRIDES,
                      constrain_ending=CONSTRAIN_ENDING)
    df.to_csv(LINKS_FILE, index=False)
    print(f'  Wrote {len(df)} links to {LINKS_FILE}')

# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------
calvin = CALVINCap(
    linksfile=LINKS_FILE,
    facilities_csv=CAP.get('facilities_csv') or None,
    profiles_csv=CAP.get('profiles_csv') or None,
    legacy_desal=CAP.get('legacy_desal', 'existing'),
    enforce_alpha=CAP.get('enforce_alpha', True),
    discount_rate=CAP.get('discount_rate', 0.04),
    facility_life=CAP.get('facility_life', 30),
    cap_groups=CAP.get('cap_groups') or None,
    expansions_csv=CAP.get('expansions_csv') or None,
    expansion_arcs_csv=CAP.get('expansion_arcs_csv') or None,
    exp_groups=CAP.get('exp_groups') or None,
    scenario=SCENARIO,
    env_flow=cfg.get('env_flow') or None,
    logdir=MODEL_DIR,
)
if calvin.scenario_adjustments is not None:
    calvin.scenario_adjustments.to_csv(
        os.path.join(RESULT_DIR, 'scenario_adjustments.csv'))
calvin.create_highs_model(debug_mode=False)
try:
    solved = calvin.solve_highs_model(solver=SOLVER, nproc=NPROC,
                                      debug_mode=False,
                                      solver_options=SOLVER_OPTIONS)
except (ValueError, RuntimeError):
    # stale/mismatched basis file: retry cold (HiGHS usually ignores a bad
    # basis, but guard the read_basis_file path anyway).
    if 'read_basis_file' not in SOLVER_OPTIONS:
        raise
    print('warm start failed (basis does not match this model); '
          'retrying cold')
    SOLVER_OPTIONS.pop('read_basis_file')
    solved = calvin.solve_highs_model(solver=SOLVER, nproc=NPROC,
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

capacity, cap_duals = calvin.capacity_results()
expansions, exp_duals = calvin.expansion_results()
capital_kperyr = capacity.capital_cost_kperyr.sum()
if expansions is not None:
    capital_kperyr += expansions.capital_cost_kperyr.sum()

print('total cost={} $M/yr (incl. capital)'.format(
    round((short_cost + op_cost) / 82 / 1e3 + capital_kperyr / 1e3, 2)))
print('short cost={} $M/yr'.format(round(short_cost / 82 / 1e3, 2)))
print('op cost={} $M/yr'.format(round(op_cost / 82 / 1e3, 2)))
print('capital cost={} $M/yr'.format(round(capital_kperyr / 1e3, 2)))

built = capacity[capacity.xcap_tafm > 0.001]
print('\nBuilt facilities ({} of {}):'.format(len(built), len(capacity)))
if len(built):
    print(built[['xcap_tafm', 'annual_tafy', 'xcap_max_tafm',
                 'capital_cost_kperyr', 'mv_capacity_kperyr',
                 'balance_gap']].to_string())

if expansions is not None:
    exp_built = expansions[expansions.xexp > 0.001]
    print('\nBuilt expansions ({} of {}):'.format(len(exp_built),
                                                  len(expansions)))
    if len(exp_built):
        print(exp_built[['xexp', 'unit', 'annual_tafy', 'xexp_max',
                         'capital_cost_kperyr', 'mv_capacity_kperyr',
                         'balance_gap']].to_string())

gw_initial = model.loc[(model.index.str.contains('GW_')) & (model.index.str.contains('INITIAL'))]
gw_final   = model.loc[(model.index.str.contains('GW_')) & (model.index.str.contains('FINAL'))]
gw_change  = gw_final['flow'].values - gw_initial['lower_bound'].values
gw_od      = gw_change[np.where(gw_change < 0)].sum()
print('gw overdraft={} MAF/yr'.format(round(gw_od / 82 / 1e3, 2)))

capacity.to_csv(os.path.join(RESULT_DIR, 'capacity.csv'))
cap_duals.to_csv(os.path.join(RESULT_DIR, 'capacity_duals.csv'))
if expansions is not None:
    expansions.to_csv(os.path.join(RESULT_DIR, 'expansions.csv'))
    exp_duals.to_csv(os.path.join(RESULT_DIR, 'expansion_duals.csv'))

postprocess(calvin.df, calvin.hmodel, RESULT_DIR)
generate_report(MODEL_DIR)
