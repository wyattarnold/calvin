"""Targeted feasibility relaxation for dry-future CALVIN solves.

Under a supply-cutting future (Colorado cut + climate warm-shift) a dry
node-month can drive the LP physically infeasible: a forced floor (a required
depletion, an instream-flow minimum, a dead-pool storage bound) demands more
water than the cut inflow can supply, and mass balance cannot close. The base
repo handles this with *debug mode* — DBUGSRC/DBUGSNK valve arcs priced at a
2e10 big-M plus a bound-relaxing heuristic — which flattens real costs, scatters
untargeted bound edits, and reports a meaningless penalty volume.

This module replaces that with a **targeted elastic relaxation** of a
pre-catalogued set of hard lower-bound floors:

1. :func:`relaxable_floors` catalogs every ``lower_bound > 0`` arc that could go
   infeasible in a drought, classified by topological signature (there is no
   property flag in the network data — category is encoded by the bound type and
   the arc's endpoints). Forced rim inflows and initial storage are the scenario
   givens and are excluded.
2. :func:`add_relaxation` makes each catalogued floor *elastic*: it frees the
   arc's lower bound and adds a non-negative shortfall slack ``s`` tied by a row
   ``arc_flow + s >= floor``. Where water allows the floor is met (``s = 0``);
   where it can't, ``s`` absorbs exactly the minimum deficit.
3. :func:`solve_two_phase` first tries a plain solve (zero overhead when the
   future is feasible); on infeasibility it runs the elastic model in two
   lexicographic phases — phase 1 minimizes the weighted shortfall, phase 2
   fixes that shortfall and re-optimizes the real economic operation — so the
   flows stay cost-meaningful and the nonzero slacks are a physical
   unmet-obligation report.

The per-category :data:`CATEGORY_WEIGHTS` set which obligation breaks first: the
CALSIM GW-SW depletion terms and accounting splits (calibration artifacts, not
policy) absorb the drought first; environmental and regulatory flows are the
most protected, so their shortfall is the meaningful cost-of-inaction signal.

If the fully-elastic model is *still* infeasible, the conflict lives outside the
catalog (a routing capacity, an inflow/initial-storage inconsistency); the driver
logs the native IIS and raises rather than papering over it — so the elastic set
doubles as a proof that every relaxable floor was enumerated.

HiGHS-only, by the two-stage-cap thread convention. Sibling to
``calvin/env_flow.py`` (same backend-neutral row/column spec pattern).
"""
from collections import namedtuple

import numpy as np
import pandas as pd

# Boundary nodes carry no timestamp and no mass-balance row.
BOUNDARY = frozenset(['SOURCE', 'INFLOW', 'INITIAL', 'FINAL', 'SINK',
                      'OUTBOUND', 'INBOUND', 'DBUGSRC', 'DBUGSNK'])

# --- categories (also the report labels) ---------------------------------
CAT_DEPLETION = 'depletion_sink'          # CALSIM GW-SW accretion/depletion
CAT_REQ_DELTA = 'required_delta_outflow'  # D-1641 regulatory
CAT_MINFLOW = 'min_instream_flow'         # environmental river-reach floor
CAT_TRANSFER = 'forced_transfer'          # refuge/urban-split/bypass EQT
CAT_DEADPOOL = 'dead_pool_storage'        # reservoir operating minimum
CAT_EOP = 'end_of_period_storage'         # forced ending-storage target

# Phase-1 penalty per category: which obligation breaks first (cheap) vs is
# protected (expensive). Widely separated so the ordering is lexicographic in
# practice, but phase 1 still minimizes the total weighted TAF relaxed.
# Rationale (user decision "calibration first, protect regs"): depletion and
# accounting splits are calibration artifacts and absorb the drought first;
# environmental/regulatory flows are the meaningful cost.
CATEGORY_WEIGHTS = {
    CAT_DEPLETION: 1.0,
    CAT_TRANSFER: 1.0,
    CAT_DEADPOOL: 10.0,
    CAT_EOP: 10.0,
    CAT_MINFLOW: 1_000.0,
    CAT_REQ_DELTA: 10_000.0,
}

# A single relaxable floor: the arc, its floor value, category, and the
# responsible node + month (for the report).
Floor = namedtuple('Floor', ['arc', 'l', 'category', 'node', 'month'])

# Handle returned by add_relaxation, consumed by solve_two_phase / report.
RelaxHandle = namedtuple('RelaxHandle', ['floors', 'slack_meta', 'slack_keys',
                                         'weights'])

# What solve_two_phase returns.
RelaxSolution = namedtuple('RelaxSolution',
                           ['relaxed', 'total_taf', 'weighted_relaxation',
                            'objective', 'report'])


def _base(node):
  """Strip the ``.YYYY-MM-DD`` timestamp from a node id."""
  s = str(node)
  dot = s.find('.')
  return s if dot < 0 else s[:dot]


def _month_label(*nodes):
  """First ``YYYY-MM`` found among the (timestamped) endpoints, else ''."""
  for node in nodes:
    s = str(node)
    dot = s.find('.')
    if dot >= 0 and len(s) >= dot + 8:
      return s[dot + 1:dot + 8]
  return ''


def _slack_key(arc):
  """Registered column key for an arc's shortfall slack (disjoint from arc keys)."""
  return ('S', arc[0], arc[1], arc[2])


def _classify(ib, jb, lb, ub):
  """Category for a ``lb>0`` arc, or None to skip (given / not a floor).

  First match wins; order matters (SINK/FINAL/Req_Delta before the generic
  internal-internal test, and the givens excluded first)."""
  if ib in ('INFLOW', 'INITIAL', 'DBUGSRC') or jb == 'DBUGSNK':
    return None                                   # scenario givens / debug
  if jb == 'FINAL':
    return CAT_EOP
  if jb == 'SINK':
    return CAT_DEPLETION
  if jb == 'Req_Delta':
    return CAT_REQ_DELTA
  if ib == jb:
    return CAT_DEADPOOL                            # storage self-loop carryover
  if ib not in BOUNDARY and jb not in BOUNDARY:
    return CAT_MINFLOW if lb < ub else CAT_TRANSFER
  return None


def relaxable_floors(df):
  """Catalog the hard lower-bound floors that a drought could make infeasible.

  Runs on the **final** links df — after futures + ``scenario.apply_delivery_floors``
  + env-flow — so any injected institutional floor is catalogued too. Excludes
  the forced rim inflows and initial storage (the scenario givens). Over-inclusion
  is harmless: a floor that never binds gets a zero slack and stays fully enforced.

  :returns: list of :class:`Floor`.
  """
  sub = df[df['lower_bound'] > 0]
  if sub.empty:
    return []
  i = sub['i'].to_numpy()
  j = sub['j'].to_numpy()
  k = sub['k'].to_numpy()
  lb = sub['lower_bound'].to_numpy(dtype=float)
  ub = sub['upper_bound'].to_numpy(dtype=float)
  ib = np.array([_base(x) for x in i])
  jb = np.array([_base(x) for x in j])

  floors = []
  for n in range(len(sub)):
    cat = _classify(ib[n], jb[n], lb[n], ub[n])
    if cat is None:
      continue
    arc = (i[n], j[n], k[n])
    node = str(ib[n]) if jb[n] in ('SINK', 'FINAL', 'Req_Delta') or ib[n] == jb[n] \
        else '%s-%s' % (ib[n], jb[n])
    floors.append(Floor(arc=arc, l=float(lb[n]), category=cat, node=node,
                        month=_month_label(i[n], j[n])))
  return floors


def catalog_summary(floors):
  """{category: (count, total_floor_taf)} — a quick sanity view of a catalog."""
  out = {}
  for f in floors:
    n, taf = out.get(f.category, (0, 0.0))
    out[f.category] = (n + 1, taf + f.l)
  return out


def add_relaxation(m, floors, weights=None):
  """Make each catalogued floor elastic on the persistent HiGHS model ``m``.

  For every floor arc ``x`` with lower bound ``l``: free its lower bound
  (``0 <= x <= u``), add a shortfall slack ``s`` in ``[0, l]``, and add a row
  ``x + s >= l``. The slack's initial objective coefficient is its category
  weight (phase-1 ready).

  :returns: :class:`RelaxHandle`.
  """
  weights = weights or CATEGORY_WEIGHTS
  if not floors:
    return RelaxHandle(floors=[], slack_meta={}, slack_keys=[], weights=weights)

  slack_meta = {}
  col_specs = []
  for f in floors:
    key = _slack_key(f.arc)
    slack_meta[key] = f
    col_specs.append((key, 0.0, f.l, float(weights.get(f.category, 1.0))))
  m.add_columns(col_specs)

  # free the floors (batch): lower -> 0, upper unchanged
  bound_map = {f.arc: (0.0, m.col_upper[m.arc_index[f.arc]]) for f in floors}
  m.set_bounds(bound_map)

  # x + s >= l, one row per floor
  row_specs = [({f.arc: 1.0, _slack_key(f.arc): 1.0}, '>=', f.l,
                ('relax',) + f.arc) for f in floors]
  m.add_rows(row_specs)

  return RelaxHandle(floors=list(floors), slack_meta=slack_meta,
                     slack_keys=list(slack_meta), weights=weights)


def relaxation_report(m, handle):
  """Per-node-month, per-category table of the nonzero shortfalls (the drought
  deliverable). Includes each relaxed floor's row dual — the marginal value of
  the obligation in the economic operation (the Benders feasibility-cut source).
  """
  cols = ['node', 'month', 'category', 'i', 'j', 'floor_taf', 'shortfall_taf',
          'shortfall_dual']
  if not handle.slack_keys:
    return pd.DataFrame(columns=cols)
  vals = m.cap_values()
  duals = m.coupling_duals()
  rows = []
  for key in handle.slack_keys:
    s = float(vals.get(key, 0.0))
    if s <= 1e-6:
      continue
    f = handle.slack_meta[key]
    dual = float(duals.get(('relax',) + f.arc, 0.0))
    rows.append((f.node, f.month, f.category, f.arc[0], f.arc[1], f.l, s, dual))
  out = pd.DataFrame(rows, columns=cols)
  return out.sort_values('shortfall_taf', ascending=False).reset_index(drop=True)


def solve_two_phase(m, floors, weights=None, options=None, log=None):
  """Solve ``m``, relaxing catalogued floors only if it is infeasible.

  1. Plain solve. If optimal, return with an empty report (zero overhead — the
     elastic layer is inert when the future is already feasible).
  2. Otherwise apply :func:`add_relaxation` and solve two lexicographic phases:
     phase 1 minimizes the weighted shortfall (real arc costs zeroed); phase 2
     fixes each slack at its phase-1 value, restores the real costs, and
     re-optimizes the economic operation on the minimally-relaxed network.
  3. If phase 1 is itself infeasible, the conflict is outside the catalog — log
     the native IIS and raise.

  :param m: a built, un-solved :class:`~calvin.highs_model.HighsNetworkModel`.
  :returns: :class:`RelaxSolution`.
  """
  weights = weights or CATEGORY_WEIGHTS

  def _info(msg, *a):
    if log is not None:
      log.info(msg, *a)

  # --- attempt 1: plain solve (no relaxation) ----------------------------
  empty = RelaxHandle(floors=[], slack_meta={}, slack_keys=[], weights=weights)
  if m.solve(need_duals=True, options=options, raise_on_infeasible=False,
             log_iis=False):
    _info('relax: feasible under real costs; no relaxation applied')
    return RelaxSolution(relaxed=False, total_taf=0.0, weighted_relaxation=0.0,
                         objective=m.objective(),
                         report=relaxation_report(m, empty))

  _info('relax: infeasible under real costs; applying targeted relaxation '
        'over %d catalogued floors', len(floors))
  handle = add_relaxation(m, floors, weights)

  n = m.n_arc_cols
  base_idx = np.arange(n, dtype=np.int32)
  slack_idx = np.array([m.extra_cols[k] for k in handle.slack_keys],
                       dtype=np.int32)
  slack_w = np.array([weights.get(handle.slack_meta[k].category, 1.0)
                      for k in handle.slack_keys], dtype=float)

  # --- phase 1: minimize weighted shortfall (zero the real arc costs) -----
  m.set_col_costs(base_idx, np.zeros(n))
  m.set_col_costs(slack_idx, slack_w)          # (already set at add; explicit)
  if not m.solve(need_duals=True, options=options, raise_on_infeasible=False,
                 log_iis=False):
    m.set_col_costs(base_idx, m.col_cost)      # restore before bailing
    m._log_iis()
    raise RuntimeError('relaxation infeasible: the conflict lies outside the '
                       'catalogued floors (routing capacity or inflow/initial '
                       'inconsistency) — see the IIS above')
  weighted_R = m.objective()                   # = Sigma w*s (base costs are 0)
  caps = m.cap_values()
  s1 = np.array([caps.get(k, 0.0) for k in handle.slack_keys], dtype=float)
  total_taf = float(s1.sum())
  _info('relax: phase-1 minimal relaxation = %.3f TAF (%.3g weighted) across '
        '%d node-months', total_taf, weighted_R, int((s1 > 1e-6).sum()))

  # --- phase 2: fix the shortfall, restore costs, re-optimize operation ----
  m.set_col_bounds(slack_idx, s1, s1)          # freeze the phase-1 allocation
  m.set_col_costs(slack_idx, np.zeros(len(slack_idx)))
  m.set_col_costs(base_idx, m.col_cost)        # restore real economic costs
  m.solve(need_duals=True, options=options, raise_on_infeasible=True,
          log_iis=True)

  report = relaxation_report(m, handle)
  return RelaxSolution(relaxed=True, total_taf=total_taf,
                       weighted_relaxation=float(weighted_R),
                       objective=m.objective(), report=report)
