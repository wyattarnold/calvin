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
from time import perf_counter

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
CAT_DELIVERY = 'delivery_floor'           # institutional ag/urban delivery floor

# Phase-1 penalty per category: which obligation breaks first (cheap) vs is
# protected (expensive). Widely separated so the ordering is lexicographic in
# practice, but phase 1 still minimizes the total weighted TAF relaxed.
# Rationale (user decision "calibration first, protect regs"): CALSIM depletions
# and accounting splits are calibration artifacts, so they absorb the drought
# first; a contract delivery deficiency is the measurable cost-of-inaction
# damage, so it absorbs next, before any physical storage or regulatory floor is
# touched; environmental/regulatory flows are the most protected and their
# shortfall is the meaningful signal.
CATEGORY_WEIGHTS = {
    CAT_DEPLETION: 1.0,
    CAT_TRANSFER: 1.0,
    CAT_DELIVERY: 2.0,
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


def relaxable_floors(df, delivery_links=None):
  """Catalog the hard lower-bound floors that a drought could make infeasible.

  Runs on the **final** links df — after futures + ``scenario.apply_delivery_floors``
  + env-flow — so any injected institutional floor is catalogued too. Excludes
  the forced rim inflows and initial storage (the scenario givens). Over-inclusion
  is harmless: a floor that never binds gets a zero slack and stays fully enforced.

  :param delivery_links: optional set of base ``'i-j'`` link names (from
    ``scenario.demand_links``) that the institutional layer floored. A floor on
    one of these is reclassified from the generic internal min-flow class to
    :data:`CAT_DELIVERY`, so a contract deficiency is priced and reported apart
    from a true environmental instream-flow floor. Leave ``None`` for a raw
    (market) network with no injected delivery floors.
  :returns: list of :class:`Floor`.
  """
  sub = df[df['lower_bound'] > 0]
  if sub.empty:
    return []
  delivery_links = set(delivery_links) if delivery_links else None
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
    ij = '%s-%s' % (ib[n], jb[n])
    if delivery_links is not None and cat in (CAT_MINFLOW, CAT_TRANSFER) \
        and ij in delivery_links:
      cat = CAT_DELIVERY
    arc = (i[n], j[n], k[n])
    node = str(ib[n]) if jb[n] in ('SINK', 'FINAL', 'Req_Delta') or ib[n] == jb[n] \
        else ij
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


def relax_solve_persistent(m, handle, weights=None, options=None, log=None,
                           reset_on_infeasible=True):
  """One cell's two-phase solve on a model that ALREADY carries the elastic
  scaffolding (``add_relaxation`` was called once up front).

  Unlike :func:`solve_two_phase` — which builds and tears the relaxation per
  call — this assumes a persistent, warm-started model reused across a whole
  grid: the caller has already reset the arc bounds to this cell's future +
  institutional values (``set_arc_bounds``), leaving the intrinsic floor arcs
  freed and their ``x + s >= l`` rows in place. Here we reset the slacks to a
  clean hard-floor state (fixed at 0), try a plain economic solve warm from the
  previous cell's basis, and only on infeasibility run the two lexicographic
  phases. Same minimal-relaxation semantics as :func:`solve_two_phase`, but the
  model — and its basis — carry over, so every cell after the first warm-restarts.

  :param handle: the :class:`RelaxHandle` returned by the one-time
    ``add_relaxation`` (its slack set is the fixed intrinsic-floor catalog).
  :param reset_on_infeasible: when the warm plain attempt proves infeasibility by
    dual simplex, it leaves the basis at an infeasible vertex — a poison pill that
    makes phase 1 grind through millions of degenerate pivots (>13 min, vs ~90s
    from a clean start; measured). Default True calls ``clear_basis`` after an
    infeasible plain attempt so phase 1 starts cold instead of warm-starting off
    that basis. (A ``presolve=on`` pass does NOT fix it — presolve detects the
    infeasibility but leaves the loaded basis in place, so phase 1 still inherits
    it.) Feasible cells never hit this and keep their warm-start. Set False for
    the pure-warm baseline (kept for diagnosis).
  :returns: :class:`RelaxSolution`.
  """
  weights = weights or CATEGORY_WEIGHTS
  options = dict(options or {})

  def _info(msg, *a):
    if log is not None:
      log.info(msg, *a)

  empty = RelaxHandle(floors=[], slack_meta={}, slack_keys=[], weights=weights)
  if not handle.slack_keys:                       # no floors catalogued
    m.solve(need_duals=True, options=options, raise_on_infeasible=True)
    return RelaxSolution(relaxed=False, total_taf=0.0, weighted_relaxation=0.0,
                         objective=m.objective(),
                         report=relaxation_report(m, empty))

  slack_idx = np.array([m.extra_cols[k] for k in handle.slack_keys],
                       dtype=np.int32)
  slack_w = np.array([weights.get(handle.slack_meta[k].category, 1.0)
                      for k in handle.slack_keys], dtype=float)
  floor_l = np.array([handle.slack_meta[k].l for k in handle.slack_keys],
                     dtype=float)
  z = np.zeros(len(slack_idx))
  n = m.n_arc_cols
  base_idx = np.arange(n, dtype=np.int32)

  # --- clean start: hard floors (slacks pinned to 0, unpriced) -------------
  m.set_col_bounds(slack_idx, z, z)
  m.set_col_costs(slack_idx, z)

  # --- attempt 1: plain economic solve (warm) -----------------------------
  t0 = perf_counter()
  feas = m.solve(need_duals=True, options=options, raise_on_infeasible=False,
                 log_iis=False)
  _info('relax: plain attempt %.0fs (%s)', perf_counter() - t0,
        'feasible' if feas else 'infeasible')
  if feas:
    _info('relax: feasible under real costs; no relaxation applied')
    return RelaxSolution(relaxed=False, total_taf=0.0, weighted_relaxation=0.0,
                         objective=m.objective(),
                         report=relaxation_report(m, empty))

  _info('relax: infeasible under real costs; targeted relaxation over %d '
        'catalogued floors', len(handle.slack_keys))

  # --- drop the poison basis the infeasible simplex left behind -----------
  # Clearing the retained basis makes phase 1 start cold instead of warm-starting
  # off the infeasible vertex the plain attempt parked at.
  if reset_on_infeasible:
    m.clear_basis()
    _info('relax: cleared the infeasible-plain basis before phase 1')

  # --- phase 1: minimize weighted shortfall (free slacks, zero arc costs) ---
  m.set_col_bounds(slack_idx, z, floor_l)
  m.set_col_costs(base_idx, np.zeros(n))
  m.set_col_costs(slack_idx, slack_w)
  t0 = perf_counter()
  if not m.solve(need_duals=True, options=options, raise_on_infeasible=False,
                 log_iis=False):
    m.set_col_costs(base_idx, m.col_cost)
    m._log_iis()
    raise RuntimeError('relaxation infeasible: the conflict lies outside the '
                       'catalogued floors (routing capacity or a delivery floor '
                       'held hard) — see the IIS above')
  t_p1 = perf_counter() - t0
  weighted_R = m.objective()
  caps = m.cap_values()
  s1 = np.array([caps.get(k, 0.0) for k in handle.slack_keys], dtype=float)
  total_taf = float(s1.sum())
  _info('relax: phase-1 %.0fs, minimal relaxation = %.3f TAF (%.3g weighted) '
        'across %d node-months', t_p1, total_taf, weighted_R,
        int((s1 > 1e-6).sum()))

  # --- phase 2: fix the shortfall, restore costs, re-optimize operation ----
  m.set_col_bounds(slack_idx, s1, s1)
  m.set_col_costs(slack_idx, z)
  m.set_col_costs(base_idx, m.col_cost)
  t0 = perf_counter()
  m.solve(need_duals=True, options=options, raise_on_infeasible=True,
          log_iis=True)
  _info('relax: phase-2 %.0fs', perf_counter() - t0)

  return RelaxSolution(relaxed=True, total_taf=total_taf,
                       weighted_relaxation=float(weighted_R),
                       objective=m.objective(),
                       report=relaxation_report(m, handle))
