"""Extensive-form CVaR coupler for the two-stage capacity phase (small-S).

Assembles S operational blocks (one per stochastic sample) into a single HiGHS
model that shares one first-stage build (the ``('cap', fac)`` columns), with a
Rockafellar-Uryasev CVaR objective:

    min  capital(X_cap)  +  (1-lam)/S * sum_s Q_s  +  lam*( eta + 1/((1-beta)S) sum_s z_s )
    s.t. z_s >= Q_s - eta,  z_s >= 0,  and the full operational LP of each block s,
         every block's facility arcs coupled to the SHARED X_cap.

Each block is a namespaced copy of the base network (node ``X`` -> ``X@s{s}``) so
blocks share nothing except the capacity columns and the CVaR variables. The
per-block operational + penalty cost is aggregated into an explicit ``('Q', s)``
column via a defining row ``Q_s = sum_c cost_c * col_c`` (block arcs at their real
cost; env / trade-overflow / floor slacks at a big-M penalty), so the CVaR rows can
reference ``Q_s`` directly. Feasibility is single-phase priced-soft (no relax
two-phase): every block carries penalized slacks, so the whole thing is one LP.

Small-S / reduced-horizon only (the coupled LP is ~S x a single solve). Production
scale is Benders' job (deferred); this validates the CVaR formulation and yields
the here-and-now build (RP) the wait-and-see grid compares against.

Facilities-only coupling for now (expansions disabled); the expansion registry is
symmetric and can be added the same way.
"""
import os

import numpy as np
import pandas as pd

from calvin import CALVINCap
from calvin.highs_model import HighsNetworkModel, NO_BALANCE
from calvin.futures import apply_futures
from calvin.scenario import apply_gw_export_closures
from calvin.env_flow import env_flow_rows
from calvin.trade import trade_budget_rows
from calvin.relax import relaxable_floors


# SOURCE/SINK are the whole-network boundary (no mass-balance row); keep them
# shared and un-namespaced so blocks stay independent without duplicating them.
def _ns(name, s):
  return name if name in NO_BALANCE else '%s@s%d' % (name, s)


def _ns_df(df, s):
  out = df.copy()
  suf = '@s%d' % s
  ns = lambda x: x if x in NO_BALANCE else x + suf
  out['i'] = out['i'].astype(str).map(ns)
  out['j'] = out['j'].astype(str).map(ns)
  return out


def _ns_arc(arc, s):
  return (_ns(arc[0], s), _ns(arc[1], s), arc[2])


def _prep_block(links, sample, catalog, enforce_alpha, log):
  """Construct one block's CALVINCap (df + registries), apply its sample + the
  institutional GW export closures. No model is built."""
  cal = CALVINCap(links, scenario=None, enforce_alpha=enforce_alpha,
                  expansions_csv=None, **catalog)
  apply_futures(cal.df, sample, log=log)
  apply_gw_export_closures(cal.df, log=log)
  return cal


def build_ef(links, samples, T, lam, beta, *, env_flow, catalog,
             enforce_alpha=True, penalty=None, share_build=True, log=None):
  """Assemble the S-block extensive-form CVaR model. Returns (m, meta)."""
  S = len(samples)
  cals = [_prep_block(links, sm, catalog, enforce_alpha, log) for sm in samples]
  cal0 = cals[0]
  if penalty is None:
    penalty = 10.0 * float(cal0.df['cost'].abs().max())
  penalty = float(penalty)

  fac_names = list(cal0.facilities.index)
  alpha = cal0.facilities.alpha
  cap_coeff = cal0._cap_coeff
  xcap_max = cal0._xcap_max
  fac_arcs = cal0._fac_arcs
  arc_coeff = cal0._arc_coeff
  cap_groups = getattr(cal0, 'cap_groups', {}) or {}

  m = HighsNetworkModel(log=log)
  block_costs = {}        # s -> {namespaced arc key: real cost} for the Q_s row
  block_arc_ids = []      # all block arc column indices (to zero their obj coeff)
  ents = []               # per-block {wy: entitlement}

  # -- 1. build every block's network (namespaced), merge the arc index -------
  floor_records = []      # per block: list of (ns_arc, l)
  for s, cal in enumerate(cals):
    df = cal.df
    floors = relaxable_floors(df)
    # free the floor lower bounds in the df copy; re-imposed as priced-soft rows
    frecs = []
    if floors:
      fset = {f.arc for f in floors}
      key = list(zip(df.i, df.j, df.k))
      mask = np.array([k in fset for k in key], dtype=bool)
      df = df.copy()
      df.loc[mask, 'lower_bound'] = 0.0
      frecs = [(_ns_arc(f.arc, s), float(f.l)) for f in floors]
    floor_records.append(frecs)

    ns_df = _ns_df(df, s)
    nodes = pd.unique(pd.concat([ns_df.i, ns_df.j]))
    if s == 0:
      m.build(ns_df, nodes, debug_mode=False)
      aidx = dict(m.arc_index)
    else:
      aidx, _ = m.add_block(ns_df, nodes, debug_mode=False)
      m.arc_index.update(aidx)
    # record real costs for the Q_s defining row (nonzero-cost arcs only)
    cost = df.cost.to_numpy(dtype=float)
    bc = {}
    for a, c in zip(zip(ns_df.i, ns_df.j, ns_df.k), cost):
      block_arc_ids.append(aidx[a])
      if c != 0.0:
        bc[a] = c
    block_costs[s] = bc

  # zero every block arc's objective coefficient (cost routed through Q_s)
  m.set_col_costs(np.array(block_arc_ids, dtype=np.int32),
                  np.zeros(len(block_arc_ids)))

  # -- 2. shared first-stage capacity columns --------------------------------
  if share_build:
    m.add_columns([(('cap', f), 0.0, xcap_max[f], cap_coeff[f])
                   for f in fac_names])
  else:                    # decomposition mode: per-block cap, capital / S
    for s in range(S):
      m.add_columns([(('cap', s, f), 0.0, xcap_max[f], cap_coeff[f] / S)
                     for f in fac_names])

  def capkey(s, f):
    return ('cap', f) if share_build else ('cap', s, f)

  # -- 3. per-block side constraints + Q_s ------------------------------------
  q_expected_coeff = (1.0 - lam) / S
  for s, cal in enumerate(cals):
    df = cal.df
    qcoeffs = dict(block_costs[s])                  # Q_s defining row: block arcs

    # env-flow (soft): rows + penalized slacks
    _, env_specs = env_flow_rows(df, env_flow)
    env_cols, env_rows = [], []
    for (coeffs, sense, rhs, label) in env_specs:
      skey = ('env_slack', s) + label
      env_cols.append((skey, 0.0, rhs, 0.0))       # cost via Q_s
      c = {_ns_arc(a, s): v for a, v in coeffs.items()}
      c[skey] = 1.0
      env_rows.append((c, sense, rhs, ('env', s) + label))
      qcoeffs[skey] = penalty
    m.add_columns(env_cols)

    # trade budget (soft overflow)
    tcols, trows, ent = trade_budget_rows(df, {'T_tafy': T})
    ents.append(ent)
    tcol_specs, trow_specs = [], []
    for (key, lo, hi, cost) in tcols:
      nk = (key[0], s) + key[1:]                   # ('trade_budget'|'overflow', s, wy)
      tcol_specs.append((nk, lo, hi, 0.0))
      if key[0] == 'trade_overflow':
        qcoeffs[nk] = penalty
    m.add_columns(tcol_specs)
    for (coeffs, sense, rhs, label) in trows:
      # ag arcs are (i,j,k) node tuples; budget/overflow are ('trade_*', wy) keys
      c = {}
      for a, v in coeffs.items():
        if a[0] in ('trade_budget', 'trade_overflow'):
          c[(a[0], s) + a[1:]] = v
        else:
          c[_ns_arc(a, s)] = v
      trow_specs.append((c, sense, rhs, ('trade', s) + label))
    m.add_rows(env_rows)
    m.add_rows(trow_specs)

    # floor slacks (priced-soft): arc + slack >= l
    frecs = floor_records[s]
    if frecs:
      fcols, frows = [], []
      for i, (ns_arc, l) in enumerate(frecs):
        skey = ('S', s, i)
        fcols.append((skey, 0.0, l, 0.0))
        frows.append(({ns_arc: 1.0, skey: 1.0}, '>=', l, ('floor', s, i)))
        qcoeffs[skey] = penalty
      m.add_columns(fcols)
      m.add_rows(frows)

    # facility capacity coupling to the shared X_cap
    cap_rows = []
    for f in fac_names:
      a_f = float(alpha[f]) if f in alpha.index else 0.0
      for (fac, i, j, k) in [(f,) + a for a in fac_arcs[f]]:
        ns_a = _ns_arc((i, j, k), s)
        co = arc_coeff[(fac, i, j, k)]
        cap_rows.append(({ns_a: 1.0, capkey(s, f): -co}, '<=', 0.0,
                         ('cap_upper', s, fac, i, j, k)))
        if a_f > 0:
          cap_rows.append(({ns_a: 1.0, capkey(s, f): -co * a_f}, '>=', 0.0,
                           ('cap_lower', s, fac, i, j, k)))
    m.add_rows(cap_rows)

    # shared group ceilings (per block, on the shared cap columns)
    if cap_groups:
      grp = {}
      for f in fac_names:
        g = cal0.facilities.cap_group[f]
        if g:
          grp.setdefault(g, []).append(f)
      grp_rows = []
      for g, fs in grp.items():
        ceil = cap_groups[g]
        grp_rows.append(({capkey(s, f): cal0._sum_profile[f] for f in fs},
                         '<=', ceil, ('cap_group', s, g)))
      if grp_rows and (s == 0 or not share_build):
        m.add_rows(grp_rows)

    # Q_s column + defining row
    m.add_columns([(('Q', s), -np.inf, np.inf, q_expected_coeff)])
    qcoeffs[('Q', s)] = -1.0
    m.add_rows([(qcoeffs, '==', 0.0, ('Qdef', s))])

  # -- 4. CVaR variables ------------------------------------------------------
  m.add_columns([(('eta',), -np.inf, np.inf, lam)])
  for s in range(S):
    m.add_columns([(('z', s), 0.0, np.inf, lam / ((1.0 - beta) * S))])
    m.add_rows([({('z', s): 1.0, ('eta',): 1.0, ('Q', s): -1.0}, '>=', 0.0,
                 ('cvar', s))])

  meta = dict(S=S, T=T, lam=lam, beta=beta, penalty=penalty,
              fac_names=fac_names, cap_coeff=cap_coeff, ents=ents,
              share_build=share_build)
  return m, meta


def solve_ef(m, meta, options=None):
  """Solve the assembled model and extract the build, per-block Q_s, and the CVaR
  decomposition. Returns a results dict."""
  opts = {'solver': 'simplex'}
  if options:
    opts.update(options)
  m.solve(need_duals=False, options=opts, raise_on_infeasible=True)
  cv = m.cap_values()
  S, lam, beta = meta['S'], meta['lam'], meta['beta']

  if meta['share_build']:
    build = {f: cv.get(('cap', f), 0.0) for f in meta['fac_names']}
    capital = sum(build[f] * meta['cap_coeff'][f] for f in meta['fac_names'])
    builds = [build]
  else:
    builds = [{f: cv.get(('cap', s, f), 0.0) for f in meta['fac_names']}
              for s in range(S)]
    capital = sum(cv.get(('cap', s, f), 0.0) * meta['cap_coeff'][f] / S
                  for s in range(S) for f in meta['fac_names'])

  Q = np.array([cv.get(('Q', s), 0.0) for s in range(S)])
  eta = cv.get(('eta',), 0.0)
  z = np.array([cv.get(('z', s), 0.0) for s in range(S)])
  expected_Q = float(Q.mean())
  cvar = float(eta + z.sum() / ((1.0 - beta) * S))
  objective = m.objective()
  return dict(objective=objective, capital=capital, Q=Q, eta=eta, z=z,
              expected_Q=expected_Q, cvar=cvar, builds=builds,
              RP=capital + expected_Q)
