"""
Capacity-expansion extension for CALVIN (Phase 1: deterministic PF).

Loads candidate facilities from calvin/data/facilities.csv, injects their
time-expanded links into the network, and extends the Pyomo model with
first-stage capacity variables X_cap coupled to every monthly facility flow.

Conventions (full rationale in tmp/notes/facilities-mapping.md):

- X_cap[f] is in TAF/month, measured on the facility intake side.
- Monthly coupling on the facility arc (flow X measured at the terminus,
  after the arc amplitude): amp*alpha*X_cap <= X <= amp*profile_m*X_cap.
- X_cap upper limit: max_cap_tafy / sum(profile); a blank profile means flat
  availability (profile of ones, sum 12), so the limit is max_cap_tafy / 12.
- Objective units are $1000s (link costs are numerically $/AF against TAF
  flows), so the capital term per facility is
  cap_cost_per_afy * CRF * sum(profile) * n_years * X_cap.

Facilities sourced from SOURCE get an intermediate plant node FAC_<name> so
each facility arc is uniquely named; reuse facilities draw directly from
their WWP wastewater node so effluent availability stays endogenous.
"""
import os

import pandas as pd
from pyomo.environ import (Set, Var, Constraint, Objective,
                           NonNegativeReals, minimize)

from calvin.calvin import CALVIN, BASE_DIR

DEFAULT_FACILITIES_CSV = os.path.join(BASE_DIR, 'data', 'facilities.csv')
DEFAULT_PROFILES_CSV = os.path.join(BASE_DIR, 'data', 'facility_profiles.csv')

# Shared infrastructure/regulatory ceilings (TAF/yr) across cap_group members
DEFAULT_CAP_GROUPS = {'sc_effluent': 500.0, 'bay_effluent': 160.0}

# Legacy Desal node treatment: existing plants only (TAF/month)
# Carlsbad ~56 TAF/yr at San Diego, Santa Barbara ~3.1 TAF/yr at SB-SLO
LEGACY_DESAL_EXISTING = {'U511': 4.667, 'U406': 0.26}

# facility_profiles.csv columns are water-year ordered
_WY_MONTHS = {'oct': 10, 'nov': 11, 'dec': 12, 'jan': 1, 'feb': 2, 'mar': 3,
              'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9}

COST_DECIMALS = 5
BOUND_DECIMALS = 3


def crf(rate, life):
  """Capital recovery factor r(1+r)^n / ((1+r)^n - 1)."""
  if rate == 0:
    return 1.0 / life
  g = (1.0 + rate) ** life
  return rate * g / (g - 1.0)


def load_facilities(facilities_csv=None, profiles_csv=None):
  """
  Load the facility table and monthly availability profiles.

  :param facilities_csv: path to facilities CSV (default calvin/data/facilities.csv)
  :param profiles_csv: path to profiles CSV (default calvin/data/facility_profiles.csv)
  :returns: (DataFrame indexed by facility name, dict profile -> {month: fraction})
  """
  facilities_csv = facilities_csv or DEFAULT_FACILITIES_CSV
  profiles_csv = profiles_csv or DEFAULT_PROFILES_CSV

  fac = pd.read_csv(facilities_csv, dtype={'cap_group': str, 'ub_profile': str})
  fac['cap_group'] = fac['cap_group'].fillna('')
  fac['ub_profile'] = fac['ub_profile'].fillna('')
  if fac.name.duplicated().any():
    raise ValueError('Duplicate facility names in %s' % facilities_csv)
  fac.set_index('name', inplace=True)

  profs = pd.read_csv(profiles_csv).set_index('profile')
  profiles = {p: {_WY_MONTHS[m]: float(row[m]) for m in _WY_MONTHS}
              for p, row in profs.iterrows()}

  missing = set(fac.ub_profile[fac.ub_profile != '']) - set(profiles)
  if missing:
    raise ValueError('Profiles referenced but not defined: %s' % sorted(missing))
  return fac, profiles


class CALVINCap(CALVIN):
  """
  CALVIN with endogenous facility capacity (Phase 1 deterministic PF).

  Extends the network with facility links at construction time and the Pyomo
  model with X_cap variables, monthly coupling constraints, shared group
  caps, and an annualized capital cost term in the objective.
  """

  def __init__(self, linksfile, facilities_csv=None, profiles_csv=None,
               legacy_desal='existing', discount_rate=0.04, facility_life=30,
               cap_groups=None, scenario=None, enforce_alpha=True, **kwargs):
    """
    :param linksfile: (string) CSV of the time-expanded monthly network
    :param facilities_csv: (string) facility table (default calvin/data/facilities.csv)
    :param profiles_csv: (string) monthly availability profiles CSV
    :param legacy_desal: (string) treatment of the legacy Desal node arcs:
      'keep' (leave the $2,072/AF unlimited backstop), 'existing' (bound to
      existing plants: Carlsbad at U511, Santa Barbara at U406), or 'retire'
      (zero all legacy desal deliveries)
    :param discount_rate: (float) CRF discount rate for capital annualization
    :param facility_life: (int) CRF facility life in years
    :param cap_groups: (dict) group name -> shared capacity ceiling in TAF/yr
    :param scenario: (dict) institutionally constrained scenario config
      (see calvin.scenario.apply_scenario); None or enabled=False runs the
      unconstrained market baseline
    :param enforce_alpha: (bool) True (default) keeps the take-or-pay lower
      couplings (built capacity must run at >= alpha every month). False
      zeroes all alphas: capacity may idle and run only in the worst years
      (the classic CALVIN capacity-expansion assumption)
    """
    super().__init__(linksfile, **kwargs)

    self.facilities, self.profiles = load_facilities(facilities_csv, profiles_csv)
    if not enforce_alpha:
      self.facilities['alpha'] = 0.0
      self.log.info('enforce_alpha=False: alpha zeroed, capacity may idle')
    self.cap_groups = DEFAULT_CAP_GROUPS if cap_groups is None else dict(cap_groups)
    self.crf = crf(discount_rate, facility_life)

    groups_used = set(self.facilities.cap_group[self.facilities.cap_group != ''])
    undefined = groups_used - set(self.cap_groups)
    if undefined:
      raise ValueError('cap_group(s) %s have no ceiling in cap_groups'
                       % sorted(undefined))

    self._steps = self._get_monthly_steps()
    self.n_years = len(self._steps) / 12.0

    # Per-facility derived data filled by _inject_facilities()
    self._sum_profile = {}   # facility -> sum of monthly fractions
    self._xcap_max = {}      # facility -> X_cap upper limit (TAF/month)
    self._cap_coeff = {}     # facility -> capital objective coefficient ($1000s per TAF/month)
    self._fac_arcs = {}      # facility -> [(i, j, k), ...] coupled arcs in step order
    self._arc_coeff = {}     # (facility, i, j, k) -> amplitude * profile fraction

    self._inject_facilities()
    self._apply_legacy_desal(legacy_desal)

    self.scenario_adjustments = None
    if scenario and scenario.get('enabled', True):
      from calvin.scenario import apply_scenario
      self.scenario_adjustments = apply_scenario(self.df, scenario, log=self.log)

    self.nodes = pd.unique(self.df[['i', 'j']].values.ravel()).tolist()
    self.links = list(zip(self.df.i, self.df.j, self.df.k))
    self.networkcheck()
    self._initial_bounds = self.df[['lower_bound', 'upper_bound']].copy()

  def _get_monthly_steps(self):
    """Sorted list of monthly date strings taken from INFLOW node stamps."""
    inflow = self.df.i[self.df.i.str.startswith('INFLOW.')]
    steps = sorted({i.split('.', 1)[1] for i in inflow})
    if not steps:
      raise ValueError('No INFLOW.<date> nodes found; is this a monthly matrix?')
    return steps

  def _profile_fraction(self, fac_row, month):
    if fac_row.ub_profile:
      return self.profiles[fac_row.ub_profile][month]
    return 1.0

  def _inject_facilities(self):
    """Append time-expanded facility links to self.df and build arc registry."""
    rows = []
    for name, f in self.facilities.iterrows():
      sum_prof = (sum(self.profiles[f.ub_profile].values())
                  if f.ub_profile else 12.0)
      xcap_max = f.max_cap_tafy / sum_prof
      self._sum_profile[name] = sum_prof
      self._xcap_max[name] = xcap_max
      self._cap_coeff[name] = (f.cap_cost_per_afy * self.crf
                               * sum_prof * self.n_years)
      self._fac_arcs[name] = []

      op_cost = round(float(f.op_cost_per_af), COST_DECIMALS)
      amp = round(float(f.amplitude), COST_DECIMALS)

      for step in self._steps:
        month = int(step[5:7])
        pf = self._profile_fraction(f, month)
        if f.source_node == 'SOURCE':
          fac_node = 'FAC_%s.%s' % (name, step)
          i, j = fac_node, '%s.%s' % (f.dest_node, step)
          rows.append(('SOURCE', fac_node, 0, 0.0, 1.0, 0.0,
                       round(pf * xcap_max, BOUND_DECIMALS)))
        else:
          i, j = '%s.%s' % (f.source_node, step), '%s.%s' % (f.dest_node, step)
        rows.append((i, j, 0, op_cost, amp, 0.0,
                     round(amp * pf * xcap_max, BOUND_DECIMALS)))
        self._fac_arcs[name].append((i, j, 0))
        self._arc_coeff[(name, i, j, 0)] = amp * pf

    new = pd.DataFrame(rows, columns=['i', 'j', 'k', 'cost', 'amplitude',
                                      'lower_bound', 'upper_bound'])
    new['link'] = new.i + '_' + new.j + '_' + new.k.map(str)
    new.set_index('link', inplace=True)

    clash = new.index.intersection(self.df.index)
    if len(clash):
      raise ValueError('Facility links collide with existing links, e.g. %s'
                       % list(clash[:3]))
    self.df = pd.concat([self.df, new])
    self.log.info('Injected %d facility links for %d facilities'
                  % (len(new), len(self.facilities)))

  def _apply_legacy_desal(self, mode):
    """Bound the legacy Desal node's delivery arcs per the chosen mode."""
    if mode == 'keep':
      return
    if mode not in ('existing', 'retire'):
      raise ValueError("legacy_desal must be 'keep', 'existing' or 'retire'")

    ix = self.df.i.str.startswith('Desal.') & self.df.j.str.match(r'U\d')
    if mode == 'retire':
      self.df.loc[ix, 'upper_bound'] = 0.0
    else:
      hubs = self.df.loc[ix, 'j'].str.split('.').str[0]
      self.df.loc[ix, 'upper_bound'] = hubs.map(
          lambda h: LEGACY_DESAL_EXISTING.get(h, 0.0)).values
    self.log.info("Legacy Desal arcs set to '%s' (%d links)" % (mode, ix.sum()))

  def create_pyomo_model(self, **kwargs):
    """Build the standard CALVIN model, then add the capacity extension."""
    super().create_pyomo_model(**kwargs)
    self._extend_pyomo_capacity()

  def _extend_pyomo_capacity(self):
    model = self.model
    fac_names = list(self.facilities.index)

    model.F = Set(initialize=fac_names)
    model.X_cap = Var(model.F, within=NonNegativeReals,
                      bounds=lambda m, fac: (0.0, self._xcap_max[fac]))

    coupled = [(fac, i, j, k) for fac in fac_names
               for (i, j, k) in self._fac_arcs[fac]]
    model.FA = Set(initialize=coupled, dimen=4)

    arc_coeff = self._arc_coeff
    alpha = self.facilities.alpha.to_dict()

    def cap_upper_rule(m, fac, i, j, k):
      return m.X[i, j, k] <= arc_coeff[(fac, i, j, k)] * m.X_cap[fac]
    model.cap_upper = Constraint(model.FA, rule=cap_upper_rule)

    coupled_lb = [(fac, i, j, k) for (fac, i, j, k) in coupled if alpha[fac] > 0]
    model.FA_lb = Set(initialize=coupled_lb, dimen=4)

    def cap_lower_rule(m, fac, i, j, k):
      amp = arc_coeff[(fac, i, j, k)]  # amplitude * profile; profile is 1 when alpha > 0
      return m.X[i, j, k] >= amp * alpha[fac] * m.X_cap[fac]
    model.cap_lower = Constraint(model.FA_lb, rule=cap_lower_rule)

    groups = {}
    for fac, g in self.facilities.cap_group.items():
      if g:
        groups.setdefault(g, []).append(fac)
    model.G = Set(initialize=list(groups))

    def cap_group_rule(m, g):
      return (sum(self._sum_profile[fac] * m.X_cap[fac] for fac in groups[g])
              <= self.cap_groups[g])
    model.cap_group = Constraint(model.G, rule=cap_group_rule)

    cap_coeff = self._cap_coeff
    model.del_component('total')

    def obj_fxn(m):
      return (sum(m.c[i, j, k] * m.X[i, j, k] for (i, j, k) in m.A)
              + sum(cap_coeff[fac] * m.X_cap[fac] for fac in m.F))
    model.total = Objective(rule=obj_fxn, sense=minimize)

    self.log.info('Capacity extension: %d X_cap vars, %d coupling constraints,'
                  ' %d group caps' % (len(fac_names), len(coupled), len(groups)))

  def capacity_results(self):
    """
    Extract capacity decisions and duals after a solve.

    :returns: (summary DataFrame indexed by facility, monthly duals DataFrame
      dates-as-rows facilities-as-columns for the coupling upper bound)

    Summary columns: xcap_tafm (TAF/month intake capacity), annual_tafy
    (deliverable TAF/yr), capital_cost_kperyr (annualized capital, $1000s/yr),
    mv_capacity_kperyr (profile-weighted sum of upper-coupling duals,
    $1000s/yr per TAF/month), top_burden_kperyr (same for the take-or-pay
    lower bound), balance_gap (capital coefficient minus net dual value,
    ~0 at an interior optimum per the design doc §4.3 condition).
    """
    model = self.model
    dual = self.model.dual

    summary = []
    ub_duals = {}
    for fac in self.facilities.index:
      xcap = model.X_cap[fac].value or 0.0
      mv = 0.0
      top = 0.0
      monthly = {}
      for (i, j, k) in self._fac_arcs[fac]:
        step = i.split('.', 1)[1] if '.' in i else j.split('.', 1)[1]
        coeff = self._arc_coeff[(fac, i, j, k)]
        d_ub = dual.get(model.cap_upper[fac, i, j, k], 0.0) or 0.0
        monthly[step] = d_ub
        mv += coeff * abs(d_ub)
        if (fac, i, j, k) in model.FA_lb:
          d_lb = dual.get(model.cap_lower[fac, i, j, k], 0.0) or 0.0
          top += coeff * self.facilities.alpha[fac] * abs(d_lb)
      ub_duals[fac] = monthly

      per_yr = self.n_years
      summary.append({
          'facility': fac,
          'xcap_tafm': round(xcap, 3),
          'annual_tafy': round(self._sum_profile[fac] * xcap, 3),
          'xcap_max_tafm': round(self._xcap_max[fac], 3),
          'capital_cost_kperyr': round(self._cap_coeff[fac] * xcap / per_yr, 1),
          'mv_capacity_kperyr': round(mv / per_yr, 1),
          'top_burden_kperyr': round(top / per_yr, 1),
          'balance_gap': round((self._cap_coeff[fac] - mv + top) / per_yr, 1),
      })

    summary_df = pd.DataFrame(summary).set_index('facility')
    duals_df = pd.DataFrame(ub_duals)
    duals_df.index.name = 'date'
    return summary_df, duals_df.sort_index()
