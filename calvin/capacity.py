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

Arc expansions (calvin/data/expansions.csv + expansion_arcs.csv) size
capacity ON EXISTING arcs — storage raises, conveyance, canal restoration,
banking, FIRO. Design (full rationale in
tmp/notes/capacity-expansion-representations.md):

- X_exp[e] is in native units: TAF/month of added arc capacity
  (unit='flow') or TAF of added storage volume (unit='storage');
  max_expansion is the X_exp upper bound in those units.
- Never mutate existing bounds: for each coupled arc and month a NEW
  k-segment (k = max existing k + 1) is appended with lb 0, cost = the
  arc's om_var_per_af, and static ub = shape_m * coeff * max_expansion,
  clamped to static_cap - sum(existing ub) when a physical ceiling is
  given (FIRO). Coupling X <= amp*shape_m*coeff*X_exp then governs; at
  X_exp = 0 the segment is dead and the base network is unchanged.
- Capital: unit='flow' follows the facility convention
  (cap_cost_per_af * CRF * sum_shape * n_years); unit='storage' charges
  (cap_cost_per_af * CRF + om_fixed_per_af_yr) * n_years per TAF of
  volume, no shape factor. Per-expansion life_yr overrides facility_life
  in the CRF.
"""
import os

import pandas as pd
from pyomo.environ import (Set, Var, Constraint, Objective,
                           NonNegativeReals, minimize)

from calvin.calvin import CALVIN, BASE_DIR

DEFAULT_FACILITIES_CSV = os.path.join(BASE_DIR, 'data', 'facilities.csv')
DEFAULT_PROFILES_CSV = os.path.join(BASE_DIR, 'data', 'facility_profiles.csv')
DEFAULT_EXPANSIONS_CSV = os.path.join(BASE_DIR, 'data', 'expansions.csv')
DEFAULT_EXPANSION_ARCS_CSV = os.path.join(BASE_DIR, 'data',
                                          'expansion_arcs.csv')

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


def load_expansions(expansions_csv=None, arcs_csv=None):
  """
  Load the arc-expansion table and its coupled-arc list.

  :param expansions_csv: path (default calvin/data/expansions.csv)
  :param arcs_csv: path (default calvin/data/expansion_arcs.csv)
  :returns: (DataFrame indexed by expansion name, arcs DataFrame with one
    row per coupled base arc)
  """
  expansions_csv = expansions_csv or DEFAULT_EXPANSIONS_CSV
  arcs_csv = arcs_csv or DEFAULT_EXPANSION_ARCS_CSV

  exp = pd.read_csv(expansions_csv,
                    dtype={'exp_group': str, 'ub_profile': str, 'notes': str})
  for col in ('exp_group', 'ub_profile', 'notes'):
    exp[col] = exp[col].fillna('')
  if exp.name.duplicated().any():
    raise ValueError('Duplicate expansion names in %s' % expansions_csv)
  bad_unit = set(exp.unit) - {'flow', 'storage'}
  if bad_unit:
    raise ValueError("expansion unit must be 'flow' or 'storage', got %s"
                     % sorted(bad_unit))
  exp.set_index('name', inplace=True)

  arcs = pd.read_csv(arcs_csv, dtype={'ub_profile_override': str})
  arcs['ub_profile_override'] = arcs['ub_profile_override'].fillna('')
  arcs['om_var_per_af'] = arcs['om_var_per_af'].fillna(0.0)
  arcs['coeff'] = arcs['coeff'].fillna(1.0)

  # optional 'enabled' column: enabled=0 parks an expansion (and its coupled
  # arcs) out of the model without deleting the row, so it can be switched back
  # on by editing one cell. Missing column means every expansion is enabled.
  if 'enabled' in exp.columns:
    disabled = set(exp.index[exp.enabled.fillna(1).astype(int) == 0])
    if disabled:
      exp = exp[~exp.index.isin(disabled)]
      arcs = arcs[~arcs.expansion.isin(disabled)]

  orphans = set(arcs.expansion) - set(exp.index)
  if orphans:
    raise ValueError('expansion_arcs rows for unknown expansions: %s'
                     % sorted(orphans))
  armless = set(exp.index) - set(arcs.expansion)
  if armless:
    raise ValueError('Expansions with no coupled arcs: %s' % sorted(armless))
  return exp, arcs


class CALVINCap(CALVIN):
  """
  CALVIN with endogenous facility capacity (Phase 1 deterministic PF).

  Extends the network with facility links at construction time and the Pyomo
  model with X_cap variables, monthly coupling constraints, shared group
  caps, and an annualized capital cost term in the objective.
  """

  def __init__(self, linksfile, facilities_csv=None, profiles_csv=None,
               legacy_desal='existing', discount_rate=0.04, facility_life=30,
               cap_groups=None, scenario=None, enforce_alpha=True,
               expansions_csv=None, expansion_arcs_csv=None, exp_groups=None,
               env_flow=None, disable_facilities=None, trade_budget=None,
               **kwargs):
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
    :param expansions_csv: (string) arc-expansion table; None (default)
      disables expansions entirely, 'default' loads
      calvin/data/expansions.csv
    :param expansion_arcs_csv: (string) coupled-arc list for expansions;
      None/'default' loads calvin/data/expansion_arcs.csv
    :param exp_groups: (dict) group name -> shared annual ceiling (TAF/yr)
      across exp_group members (flow-unit expansions only)
    :param env_flow: (dict) Bay-Delta percent-of-unimpaired env-flow config
      (see calvin.env_flow.add_env_flow_constraints); None or enabled=False
      adds no environmental-flow side constraints
    :param disable_facilities: (iterable of str) facility names to drop from
      the option set at load time, for counterfactual runs (e.g. "no
      conservation available"). Unknown names raise; the loaded CSV is left
      untouched so one catalog serves every variant.
    """
    super().__init__(linksfile, **kwargs)

    self.facilities, self.profiles = load_facilities(facilities_csv, profiles_csv)
    if disable_facilities:
      drop = set(disable_facilities)
      unknown = drop - set(self.facilities.index)
      if unknown:
        raise ValueError('disable_facilities names not in catalog: %s'
                         % sorted(unknown))
      self.facilities = self.facilities[~self.facilities.index.isin(drop)]
      self.log.info('disable_facilities: dropped %d facilities (%s)',
                    len(drop), ', '.join(sorted(drop)))
    if not enforce_alpha:
      self.facilities['alpha'] = 0.0
      self.log.info('enforce_alpha=False: alpha zeroed, capacity may idle')
    self.cap_groups = DEFAULT_CAP_GROUPS if cap_groups is None else dict(cap_groups)
    self.discount_rate = discount_rate
    self.crf = crf(discount_rate, facility_life)

    self.expansions = None
    self.expansion_arcs = None
    self.exp_groups = dict(exp_groups) if exp_groups else {}
    if expansions_csv:
      self.expansions, self.expansion_arcs = load_expansions(
          None if expansions_csv == 'default' else expansions_csv,
          None if expansion_arcs_csv in (None, '', 'default')
          else expansion_arcs_csv)
      if not enforce_alpha:
        self.expansions['alpha'] = 0.0
      # alpha on a storage-unit expansion would force water to SIT in the
      # expanded space every month — a config error, not take-or-pay
      bad = self.expansions[(self.expansions.alpha > 0)
                            & (self.expansions.unit == 'storage')]
      if len(bad):
        raise ValueError('alpha > 0 is not supported for storage-unit '
                         'expansions: %s' % sorted(bad.index))
      multi = self.expansion_arcs.expansion.value_counts()
      for name in self.expansions.index[self.expansions.alpha > 0]:
        if multi.get(name, 0) > 1:
          self.log.warning(
              'Expansion %s: alpha > 0 forces simultaneous minimum flow on '
              'ALL %d coupled arcs (for banking-style put/take pairs this '
              'is circular pumping)' % (name, multi[name]))
      prof_refs = (set(self.expansions.ub_profile)
                   | set(self.expansion_arcs.ub_profile_override))
      missing = prof_refs - set(self.profiles) - {'', 'flat'}
      if missing:
        raise ValueError('Expansion profiles not defined: %s' % sorted(missing))
      groups_used = set(
          self.expansions.exp_group[self.expansions.exp_group != ''])
      undefined = groups_used - set(self.exp_groups)
      if undefined:
        raise ValueError('exp_group(s) %s have no ceiling in exp_groups'
                         % sorted(undefined))

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

    # Per-expansion derived data filled by _inject_expansions()
    self._exp_sum_shape = {}  # expansion -> sum of monthly shape fractions
    self._xexp_max = {}       # expansion -> X_exp upper bound (native units)
    self._exp_cap_coeff = {}  # expansion -> capital objective coefficient
    self._exp_arcs = {}       # expansion -> [(i, j, k_exp), ...]
    self._exp_coeff = {}      # (expansion, i, j, k) -> amp * shape * coeff
    self._exp_rows = pd.Index([])  # df index labels of injected segments

    self._inject_facilities()
    if self.expansions is not None:
      self._inject_expansions()
    self._check_forced_inflows()
    self._apply_legacy_desal(legacy_desal)

    self.scenario_adjustments = None
    if scenario and scenario.get('enabled', True):
      from calvin.scenario import apply_scenario
      # expansion segments are the decision variables under study, not part
      # of the existing system the scenario constrains — exclude them
      self.scenario_adjustments = apply_scenario(self.df, scenario,
                                                 log=self.log,
                                                 protect=self._exp_rows)

    self.env_flow = env_flow
    self.env_flow_req = None
    self.trade_budget = trade_budget
    self.trade_ent = None

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

  def _facility_crf(self, f):
    """Per-facility CRF: optional life_yr column overrides facility_life."""
    life = getattr(f, 'life_yr', None)
    if life is not None and not pd.isna(life):
      return crf(self.discount_rate, float(life))
    return self.crf

  def _inject_facilities(self):
    """Append time-expanded facility links to self.df and build arc registry."""
    rows = []
    next_k = {}  # (i, j) -> next free k when facilities share an arc
    for name, f in self.facilities.iterrows():
      sum_prof = (sum(self.profiles[f.ub_profile].values())
                  if f.ub_profile else 12.0)
      xcap_max = f.max_cap_tafy / sum_prof
      self._sum_profile[name] = sum_prof
      self._xcap_max[name] = xcap_max
      self._cap_coeff[name] = (f.cap_cost_per_afy * self._facility_crf(f)
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
        # two facilities may share a (source, dest) arc (e.g. onsite_la and
        # dpr_ro_la both WWP504->U504): give each its own k segment
        k_fac = next_k.get((i, j), 0)
        next_k[(i, j)] = k_fac + 1
        rows.append((i, j, k_fac, op_cost, amp, 0.0,
                     round(amp * pf * xcap_max, BOUND_DECIMALS)))
        self._fac_arcs[name].append((i, j, k_fac))
        self._arc_coeff[(name, i, j, k_fac)] = amp * pf

    new = pd.DataFrame(rows, columns=['i', 'j', 'k', 'cost', 'amplitude',
                                      'lower_bound', 'upper_bound'])
    new['link'] = new.i + '_' + new.j + '_' + new.k.map(str)
    new.set_index('link', inplace=True)
    if new.index.duplicated().any():
      raise ValueError('Duplicate facility arc labels, e.g. %s'
                       % new.index[new.index.duplicated()][:3].tolist())

    clash = new.index.intersection(self.df.index)
    if len(clash):
      raise ValueError('Facility links collide with existing links, e.g. %s'
                       % list(clash[:3]))
    self.df = pd.concat([self.df, new])
    self.log.info('Injected %d facility links for %d facilities'
                  % (len(new), len(self.facilities)))

  def _inject_expansions(self):
    """
    Append one new k-segment per (coupled arc, month) to self.df and build
    the X_exp registry. Existing rows are never modified; at X_exp = 0 every
    injected segment is dead and the network is behavior-identical.
    """
    i_base = self.df.i.str.split('.').str[0]
    j_base = self.df.j.str.split('.').str[0]
    rows = []
    next_k = {}  # (i_full, j_full) -> next free k when arcs are shared

    for name, e in self.expansions.iterrows():
      prof = self.profiles[e.ub_profile] if e.ub_profile else None
      sum_shape = sum(prof.values()) if prof else 12.0
      xexp_max = float(e.max_expansion)
      life = float(e.life_yr) if not pd.isna(e.life_yr) else None
      e_crf = crf(self.discount_rate, life) if life else self.crf
      self._exp_sum_shape[name] = sum_shape
      self._xexp_max[name] = xexp_max
      if e.unit == 'storage':
        self._exp_cap_coeff[name] = ((e.cap_cost_per_af * e_crf
                                      + e.om_fixed_per_af_yr) * self.n_years)
      else:
        self._exp_cap_coeff[name] = (e.cap_cost_per_af * e_crf
                                     * sum_shape * self.n_years)
      self._exp_arcs[name] = []

      specs = self.expansion_arcs[self.expansion_arcs.expansion == name]
      for a in specs.itertuples():
        mask = (i_base == a.i) & (j_base == a.j)
        if not mask.any():
          raise ValueError('Expansion %s: base arc %s-%s not in network'
                           % (name, a.i, a.j))
        base = self.df[mask]
        agg = base.groupby(['i', 'j'], sort=True).agg(
            kmax=('k', 'max'), ub_tot=('upper_bound', 'sum'))
        amp0 = base[base.k == 0].set_index(['i', 'j']).amplitude

        if a.ub_profile_override == 'flat':
          aprof = None
        elif a.ub_profile_override:
          aprof = self.profiles[a.ub_profile_override]
        else:
          aprof = prof
        om = round(float(a.om_var_per_af), COST_DECIMALS)
        coeff_a = float(a.coeff)
        static_cap = None if pd.isna(a.static_cap) else float(a.static_cap)

        for (i_full, j_full), r in agg.iterrows():
          month = int(i_full.split('.', 1)[1][5:7])
          shape = aprof[month] if aprof else 1.0
          amp = (float(a.amplitude_override)
                 if not pd.isna(a.amplitude_override)
                 else float(amp0.get((i_full, j_full), 1.0)))
          ub = shape * coeff_a * xexp_max
          if static_cap is not None:
            ub = min(ub, max(0.0, static_cap - r.ub_tot))
          if ub <= 0:
            continue  # no headroom / zero-shape month: skip the dead segment
          k_exp = next_k.get((i_full, j_full), int(r.kmax) + 1)
          if k_exp >= 15:
            raise ValueError('Expansion %s: arc %s-%s needs k=%d but the '
                             'model k range is 15' % (name, a.i, a.j, k_exp))
          next_k[(i_full, j_full)] = k_exp + 1
          rows.append((i_full, j_full, k_exp, om,
                       round(amp, COST_DECIMALS), 0.0,
                       round(ub, BOUND_DECIMALS)))
          self._exp_arcs[name].append((i_full, j_full, k_exp))
          self._exp_coeff[(name, i_full, j_full, k_exp)] = (
              amp * shape * coeff_a)

      if not self._exp_arcs[name]:
        raise ValueError('Expansion %s produced no live segments (all '
                         'months clamped to zero?)' % name)

    new = pd.DataFrame(rows, columns=['i', 'j', 'k', 'cost', 'amplitude',
                                      'lower_bound', 'upper_bound'])
    new['link'] = new.i + '_' + new.j + '_' + new.k.map(str)
    new.set_index('link', inplace=True)
    clash = new.index.intersection(self.df.index)
    if len(clash):
      raise ValueError('Expansion segments collide with existing links, '
                       'e.g. %s' % list(clash[:3]))
    self.df = pd.concat([self.df, new])
    self._exp_rows = new.index
    self.log.info('Injected %d expansion segments for %d expansions'
                  % (len(new), len(self.expansions)))

  def _check_forced_inflows(self):
    """
    Advisory check for alpha > 0 facilities: at full build the take-or-pay
    flow amp*alpha*profile*xcap_max must fit the destination node's
    outbound capacity net of other inflows' hard lower bounds. A shortfall
    is not an infeasibility (X_cap is a variable; the LP just builds less)
    but it caps the facility's effective size below its nominal max — log
    the implied effective ceiling so under-builds are explainable.
    """
    forced_facs = self.facilities[self.facilities.alpha > 0]
    if forced_facs.empty:
      return
    dest_nodes = set(forced_facs.dest_node)
    j_base = self.df.j.str.split('.').str[0]
    i_base = self.df.i.str.split('.').str[0]
    out_ub = (self.df[i_base.isin(dest_nodes)]
              .groupby('i').upper_bound.sum())
    in_lb = (self.df[j_base.isin(dest_nodes)]
             .groupby('j').lower_bound.sum())
    for name, f in forced_facs.iterrows():
      eff_max = self._xcap_max[name]
      n_bind = 0
      for step in self._steps:
        month = int(step[5:7])
        unit_forced = f.amplitude * f.alpha * self._profile_fraction(f, month)
        if unit_forced <= 0:
          continue
        node = '%s.%s' % (f.dest_node, step)
        headroom = out_ub.get(node, 0.0) - in_lb.get(node, 0.0)
        ceiling = max(headroom, 0.0) / unit_forced
        if ceiling < self._xcap_max[name] - 1e-6:
          n_bind += 1
          eff_max = min(eff_max, ceiling)
      if n_bind:
        self.log.warning(
            'Facility %s (alpha=%.2f): take-or-pay flow exceeds %s outbound '
            'headroom in %d months at full build; effective X_cap ceiling '
            '%.2f of %.2f TAF/month' % (name, f.alpha, f.dest_node, n_bind,
                                        eff_max, self._xcap_max[name]))

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
    if self.env_flow and self.env_flow.get('enabled'):
      from calvin.env_flow import add_env_flow_constraints
      self.env_flow_req = add_env_flow_constraints(
          self.model, self.df, self.env_flow, log=self.log)

  def create_highs_model(self, **kwargs):
    """Build the standard CALVIN model in HiGHS, then add the capacity extension
    (columns X_cap/X_exp + coupling rows + capital objective).  HiGHS-only path
    for the two-stage capacity work; the Pyomo path above is unchanged."""
    super().create_highs_model(**kwargs)
    self._extend_highs_capacity()
    if self.env_flow and self.env_flow.get('enabled'):
      from calvin.env_flow import add_env_flow_constraints
      self.env_flow_req = add_env_flow_constraints(
          self.hmodel, self.df, self.env_flow, log=self.log)
    if self.trade_budget and self.trade_budget.get('enabled', True):
      from calvin.trade import add_trade_budget
      self.trade_ent = add_trade_budget(
          self.hmodel, self.df, self.trade_budget, log=self.log)
    return self.hmodel

  def _extend_highs_capacity(self):
    """Add capacity columns and coupling rows to the HiGHS model.  Mirrors
    _extend_pyomo_capacity: X[i,j,k] <= arc_coeff*X_cap (cap_upper), the
    take-or-pay X[i,j,k] >= arc_coeff*alpha*X_cap (cap_lower), shared group
    ceilings, and the analogous expansion terms.  Capital objective enters via
    the new columns' cost coefficients (the HiGHS objective is Sum col_cost*col,
    so no objective rebuild is needed)."""
    m = self.hmodel
    fac_names = list(self.facilities.index)
    arc_coeff = self._arc_coeff
    alpha = self.facilities.alpha.to_dict()

    # columns first (rows reference them by key), capital cost on each column
    m.add_columns([(('cap', fac), 0.0, self._xcap_max[fac], self._cap_coeff[fac])
                   for fac in fac_names])

    exp_names = (list(self.expansions.index)
                 if self.expansions is not None and len(self.expansions) else [])
    if exp_names:
      m.add_columns([(('exp', e), 0.0, self._xexp_max[e], self._exp_cap_coeff[e])
                     for e in exp_names])

    rows = []
    n_couple = 0
    for fac in fac_names:
      for (i, j, k) in self._fac_arcs[fac]:
        co = arc_coeff[(fac, i, j, k)]
        rows.append(({(i, j, k): 1.0, ('cap', fac): -co}, '<=', 0.0,
                     ('cap_upper', fac, i, j, k)))
        n_couple += 1
        if alpha[fac] > 0:
          rows.append(({(i, j, k): 1.0, ('cap', fac): -co * alpha[fac]}, '>=',
                       0.0, ('cap_lower', fac, i, j, k)))

    groups = {}
    for fac, g in self.facilities.cap_group.items():
      if g:
        groups.setdefault(g, []).append(fac)
    for g, facs in groups.items():
      rows.append(({('cap', fac): self._sum_profile[fac] for fac in facs}, '<=',
                   self.cap_groups[g], ('cap_group', g)))

    if exp_names:
      exp_coeff = self._exp_coeff
      exp_alpha = self.expansions.alpha.to_dict()
      for e in exp_names:
        for (i, j, k) in self._exp_arcs[e]:
          co = exp_coeff[(e, i, j, k)]
          rows.append(({(i, j, k): 1.0, ('exp', e): -co}, '<=', 0.0,
                       ('exp_upper', e, i, j, k)))
          if exp_alpha[e] > 0:
            rows.append(({(i, j, k): 1.0, ('exp', e): -co * exp_alpha[e]}, '>=',
                         0.0, ('exp_lower', e, i, j, k)))
      egroups = {}
      for e, g in self.expansions.exp_group.items():
        if g:
          if self.expansions.unit[e] != 'flow':
            raise ValueError('exp_group ceilings support flow-unit expansions '
                             'only (%s is %s)' % (e, self.expansions.unit[e]))
          egroups.setdefault(g, []).append(e)
      for g, es in egroups.items():
        rows.append(({('exp', e): self._exp_sum_shape[e] for e in es}, '<=',
                     self.exp_groups[g], ('exp_group', g)))

    m.add_rows(rows)
    self.log.info('Capacity extension (HiGHS): %d X_cap vars, %d coupling rows,'
                  ' %d group caps' % (len(fac_names), n_couple, len(groups)))
    if exp_names:
      self.log.info('Expansion extension (HiGHS): %d X_exp vars' % len(exp_names))

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
      # arc_coeff = amplitude * profile, so a profiled alpha=1 facility
      # (e.g. conservation) is pinned to exactly its seasonal shape
      amp = arc_coeff[(fac, i, j, k)]
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

    exp_names = list(self.expansions.index) if self.expansions is not None else []
    exp_coupled = []
    if exp_names:
      model.E = Set(initialize=exp_names)
      model.X_exp = Var(model.E, within=NonNegativeReals,
                        bounds=lambda m, e: (0.0, self._xexp_max[e]))
      exp_coupled = [(e, i, j, k) for e in exp_names
                     for (i, j, k) in self._exp_arcs[e]]
      model.EA = Set(initialize=exp_coupled, dimen=4)

      exp_coeff = self._exp_coeff
      exp_alpha = self.expansions.alpha.to_dict()

      def exp_upper_rule(m, e, i, j, k):
        return m.X[i, j, k] <= exp_coeff[(e, i, j, k)] * m.X_exp[e]
      model.exp_upper = Constraint(model.EA, rule=exp_upper_rule)

      exp_lb = [(e, i, j, k) for (e, i, j, k) in exp_coupled
                if exp_alpha[e] > 0]
      model.EA_lb = Set(initialize=exp_lb, dimen=4)

      def exp_lower_rule(m, e, i, j, k):
        return m.X[i, j, k] >= exp_coeff[(e, i, j, k)] * exp_alpha[e] * m.X_exp[e]
      model.exp_lower = Constraint(model.EA_lb, rule=exp_lower_rule)

      egroups = {}
      for e, g in self.expansions.exp_group.items():
        if g:
          if self.expansions.unit[e] != 'flow':
            raise ValueError('exp_group ceilings support flow-unit '
                             'expansions only (%s is %s)'
                             % (e, self.expansions.unit[e]))
          egroups.setdefault(g, []).append(e)
      model.EG = Set(initialize=list(egroups))

      def exp_group_rule(m, g):
        return (sum(self._exp_sum_shape[e] * m.X_exp[e] for e in egroups[g])
                <= self.exp_groups[g])
      model.exp_group = Constraint(model.EG, rule=exp_group_rule)

    cap_coeff = self._cap_coeff
    exp_cap_coeff = self._exp_cap_coeff
    model.del_component('total')

    def obj_fxn(m):
      total = (sum(m.c[i, j, k] * m.X[i, j, k] for (i, j, k) in m.A)
               + sum(cap_coeff[fac] * m.X_cap[fac] for fac in m.F))
      if exp_names:
        total += sum(exp_cap_coeff[e] * m.X_exp[e] for e in m.E)
      return total
    model.total = Objective(rule=obj_fxn, sense=minimize)

    self.log.info('Capacity extension: %d X_cap vars, %d coupling constraints,'
                  ' %d group caps' % (len(fac_names), len(coupled), len(groups)))
    if exp_names:
      self.log.info('Expansion extension: %d X_exp vars, %d coupling '
                    'constraints' % (len(exp_names), len(exp_coupled)))

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
    if getattr(self, '_backend', 'pyomo') == 'highs':
      return self._capacity_results_highs()
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

  def expansion_results(self):
    """
    Extract arc-expansion decisions and duals after a solve.

    :returns: (summary DataFrame indexed by expansion, monthly duals
      DataFrame dates-as-rows expansions-as-columns for the coupling upper
      bound) — or (None, None) when no expansions are loaded

    Summary columns: xexp (native units: TAF/month of capacity for
    unit='flow', TAF of volume for unit='storage'), unit, annual_tafy
    (flow: sum_shape * xexp; storage: NaN), xexp_max,
    capital_cost_kperyr (annualized capital + fixed O&M, $1000s/yr),
    mv_capacity_kperyr (shape-weighted sum of exp_upper duals),
    balance_gap (capital coefficient minus dual value, ~0 for an interior
    build).
    """
    if self.expansions is None or not len(self.expansions):
      return None, None
    if getattr(self, '_backend', 'pyomo') == 'highs':
      return self._expansion_results_highs()
    model = self.model
    dual = self.model.dual

    summary = []
    ub_duals = {}
    for name, e in self.expansions.iterrows():
      xexp = model.X_exp[name].value or 0.0
      mv = 0.0
      monthly = {}
      for (i, j, k) in self._exp_arcs[name]:
        step = i.split('.', 1)[1]
        d_ub = dual.get(model.exp_upper[name, i, j, k], 0.0) or 0.0
        monthly[step] = monthly.get(step, 0.0) + abs(d_ub)
        mv += self._exp_coeff[(name, i, j, k)] * abs(d_ub)
      ub_duals[name] = monthly

      per_yr = self.n_years
      summary.append({
          'expansion': name,
          'xexp': round(xexp, 3),
          'unit': e.unit,
          'annual_tafy': (round(self._exp_sum_shape[name] * xexp, 3)
                          if e.unit == 'flow' else float('nan')),
          'xexp_max': round(self._xexp_max[name], 3),
          'capital_cost_kperyr': round(
              self._exp_cap_coeff[name] * xexp / per_yr, 1),
          'mv_capacity_kperyr': round(mv / per_yr, 1),
          'balance_gap': round(
              (self._exp_cap_coeff[name] - mv) / per_yr, 1),
      })

    summary_df = pd.DataFrame(summary).set_index('expansion')
    duals_df = pd.DataFrame(ub_duals)
    duals_df.index.name = 'date'
    return summary_df, duals_df.sort_index()

  # -- HiGHS-backend result extractors --------------------------------------
  def _capacity_results_highs(self):
    """capacity_results reading X_cap and coupling-row duals from the HiGHS
    model.  Coupling duals are magnitude-abs'd for mv/top just like the Pyomo
    path, so the row-dual sign convention doesn't affect the summary."""
    capvals = self.hmodel.cap_values()
    cdual = self.hmodel.coupling_duals()
    summary = []
    ub_duals = {}
    for fac in self.facilities.index:
      xcap = capvals.get(('cap', fac), 0.0) or 0.0
      mv = 0.0
      top = 0.0
      monthly = {}
      for (i, j, k) in self._fac_arcs[fac]:
        step = i.split('.', 1)[1] if '.' in i else j.split('.', 1)[1]
        coeff = self._arc_coeff[(fac, i, j, k)]
        d_ub = cdual.get(('cap_upper', fac, i, j, k), 0.0) or 0.0
        monthly[step] = d_ub
        mv += coeff * abs(d_ub)
        if self.facilities.alpha[fac] > 0:
          d_lb = cdual.get(('cap_lower', fac, i, j, k), 0.0) or 0.0
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

  def _expansion_results_highs(self):
    """expansion_results reading X_exp and exp_upper duals from the HiGHS model."""
    capvals = self.hmodel.cap_values()
    cdual = self.hmodel.coupling_duals()
    summary = []
    ub_duals = {}
    for name, e in self.expansions.iterrows():
      xexp = capvals.get(('exp', name), 0.0) or 0.0
      mv = 0.0
      monthly = {}
      for (i, j, k) in self._exp_arcs[name]:
        step = i.split('.', 1)[1]
        d_ub = cdual.get(('exp_upper', name, i, j, k), 0.0) or 0.0
        monthly[step] = monthly.get(step, 0.0) + abs(d_ub)
        mv += self._exp_coeff[(name, i, j, k)] * abs(d_ub)
      ub_duals[name] = monthly

      per_yr = self.n_years
      summary.append({
          'expansion': name,
          'xexp': round(xexp, 3),
          'unit': e.unit,
          'annual_tafy': (round(self._exp_sum_shape[name] * xexp, 3)
                          if e.unit == 'flow' else float('nan')),
          'xexp_max': round(self._xexp_max[name], 3),
          'capital_cost_kperyr': round(
              self._exp_cap_coeff[name] * xexp / per_yr, 1),
          'mv_capacity_kperyr': round(mv / per_yr, 1),
          'balance_gap': round(
              (self._exp_cap_coeff[name] - mv) / per_yr, 1),
      })

    summary_df = pd.DataFrame(summary).set_index('expansion')
    duals_df = pd.DataFrame(ub_duals)
    duals_df.index.name = 'date'
    return summary_df, duals_df.sort_index()
