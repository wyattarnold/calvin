"""Bay-Delta percent-of-unimpaired environmental-flow constraints.

Adds minimum instream flows to a CALVIN model, per period (water year, or month
when ``monthly=True``):

  per tributary t, period p:
     Σ flow(t reservoir-release reach) ≥ pct · Σ (t unimpaired)   over p
  aggregate Delta outflow, period p:
     Σ (Req_Delta + Surp_Delta) ≥ max( existing Req_Delta requirement,
                                        Σ_t pct·unimpaired_t )      over p

``pct`` is either a flat float (e.g. 0.40 every period, the two-stage study
setting) or a year-type ladder ``{'wet':0.55,'middle':0.40,'dry':0.30}`` classed
by FIXED historical terciles of Delta-watershed unimpaired inflow
(``data/delta_unimpaired_wy.csv``; annual only). The monthly form scales each
month's requirement to that month's sampled unimpaired inflow.

Unimpaired inflow is exogenous (fixed ``INFLOW`` arc bounds), so every RHS is a
constant computed from the links DataFrame; the constraints only sum
``model.X`` flow variables. The per-tributary reservoir/reach mapping is curated
in ``data/delta_tributaries.csv``; each tributary's unimpaired inflow is derived
by graph reachability (the rim inflows that drain to its terminal reservoir).

On the HiGHS backend, ``relax``/``relax_penalty`` add a penalized slack per row
so a dry period pays a penalty instead of going infeasible (feasibility backstop
and Benders feasibility-cut source).

Design: my-models/two-stage-cap/notes/01-design/cost-of-inaction-study-design.md §4.1
"""
import os
from collections import defaultdict, deque

import pandas as pd
# Pyomo is imported lazily inside the Pyomo branch of add_env_flow_constraints so
# this module (and env_flow_rows) stays importable without pyomo, for the
# direct-HiGHS path and the pyomo-free app.

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TRIB_CSV = os.path.join(BASE_DIR, 'data', 'delta_tributaries.csv')
# fixed year-type reference: total Delta-watershed unimpaired inflow per water
# year over the full historical record (columns: water_year, unimpaired_taf).
DEFAULT_UNIMP_CSV = os.path.join(BASE_DIR, 'data', 'delta_unimpaired_wy.csv')

DELTA_NODE = 'D541'                       # western Delta; only inflow is D509
OUTFLOW_TARGETS = ('Req_Delta', 'Surp_Delta')  # the only D541 -> sea arcs
REQ_TARGET = 'Req_Delta'                  # existing (D-1641) required outflow

DEFAULT_PCT = {'wet': 0.55, 'middle': 0.40, 'dry': 0.30}


def _base(node):
  """Strip the .YYYY-MM-DD timestamp from a node id."""
  return node.split('.', 1)[0]


def _water_year(node):
  """Water year (Oct-Sep) of a timestamped node id, else None."""
  parts = node.split('.', 1)
  if len(parts) != 2:
    return None
  d = parts[1]
  y, m = int(d[:4]), int(d[5:7])
  return y + 1 if m >= 10 else y


def _year_month(node):
  """'YYYY-MM' of a timestamped node id, else None (monthly period key)."""
  parts = node.split('.', 1)
  if len(parts) != 2:
    return None
  return parts[1][:7]


def load_tributaries(path=None):
  """Read the curated tributary -> (terminal reservoir, river reach) table."""
  df = pd.read_csv(path or DEFAULT_TRIB_CSV)
  return [dict(name=r.tributary, reservoir=r.reservoir, reach=r.reach)
          for r in df.itertuples()]


def _tercile_thresholds(values):
  """(dry|middle, middle|wet) cut points from a value list, by index terciles."""
  s = sorted(values)
  n = len(s)
  return s[n // 3], s[2 * n // 3]


def historical_thresholds(path=None):
  """Fixed (lo, hi) year-type cut points from the historical Delta-watershed
  unimpaired-inflow reference (``delta_unimpaired_wy.csv``).

  Returns ``None`` if the reference file is absent (callers then fall back to
  df-relative terciles). ``analysis/run_env_full_sim.py`` writes the reference
  from a full-record network build.
  """
  path = path or DEFAULT_UNIMP_CSV
  if not os.path.isfile(path):
    return None
  ref = pd.read_csv(path)
  return _tercile_thresholds(ref['unimpaired_taf'].tolist())


def delta_watershed_unimpaired(df):
  """{water_year: total Delta-watershed unimpaired inflow TAF} from a links df.

  Sums the natural inflow (INFLOW-arc upper bound) of every rim reservoir that
  drains to the Delta (``D541``), per water year. Build this over the full
  historical network to make the fixed year-type reference
  (``data/delta_unimpaired_wy.csv``).
  """
  bi = df['i'].map(_base)
  bj = df['j'].map(_base)
  rev = defaultdict(set)
  for a, b in zip(bi.values, bj.values):
    rev[b].add(a)
  is_inflow = bi.eq('INFLOW')
  rim_delta = set(bj[is_inflow]) & _ancestors(rev, DELTA_NODE)
  out = defaultdict(float)
  for row in df[is_inflow].itertuples():
    if _base(row.j) in rim_delta:
      wy = _water_year(row.j)
      if wy is not None:
        out[wy] += row.upper_bound
  return dict(out)


def _ancestors(rev, target):
  """All base nodes from which ``target`` is reachable (BFS on predecessors)."""
  seen = {target}
  q = deque([target])
  while q:
    n = q.popleft()
    for p in rev.get(n, ()):
      if p not in seen:
        seen.add(p)
        q.append(p)
  return seen


def compute_requirements(df, tribs, pct=None, thresholds=None, monthly=False,
                         log=None):
  """Compute all env-flow RHS constants and the arcs each constraint sums.

  :param pct: either a **float** (flat percent-of-unimpaired applied to every
    period) or a **dict** ``{'wet','middle','dry'}`` (year-type ladder; annual
    only). Default is the ladder :data:`DEFAULT_PCT`.
  :param monthly: if True, one constraint per (tributary, ``'YYYY-MM'``) scaled
    to that month's unimpaired inflow; else per (tributary, water year). Because
    the sampled climate sets the ``INFLOW`` bounds, ``pct x unimpaired``
    auto-scales per sample either way.
  :param thresholds: (lo, hi) fixed year-type cut points on total Delta-watershed
    unimpaired inflow (ladder pct only). Defaults to :func:`historical_thresholds`
    so a water year gets the SAME label in any window/bootstrap sample.
  :returns: dict with keys
    ``trib`` -> list of (name, key, rhs, [arc keys]) per (tributary, period)
    ``agg``  -> list of (key, rhs, [arc keys]) per period
    ``year_type`` -> {wy: 'wet'|'middle'|'dry'} (empty for flat/monthly)
    where ``key`` is a water year (int) or a ``'YYYY-MM'`` string.
  """
  pct = DEFAULT_PCT if pct is None else pct
  flat = not isinstance(pct, dict)
  if not flat and monthly:
    raise ValueError('the year-type pct ladder is annual only; pass a flat '
                     'float pct for monthly env-flow')
  keyfn = _year_month if monthly else _water_year

  # base directed graph (predecessors) for reachability
  bi = df['i'].map(_base)
  bj = df['j'].map(_base)
  rev = defaultdict(set)
  for a, b in zip(bi.values, bj.values):
    rev[b].add(a)

  # rim inflow arcs: INFLOW -> node
  is_inflow = bi.eq('INFLOW')
  rim_nodes = set(bj[is_inflow])
  delta_ancestors = _ancestors(rev, DELTA_NODE)
  rim_delta = rim_nodes & delta_ancestors  # Delta-watershed rim inflows

  # INFLOW arc volumes (unimpaired = the natural inflow = fixed upper bound),
  # indexed by (base target node, period key)
  inflow_by_node = defaultdict(float)
  for row in df[is_inflow].itertuples():
    key = keyfn(row.j)
    if key is not None:
      inflow_by_node[(_base(row.j), key)] += row.upper_bound
  all_keys = sorted({k for (_, k) in inflow_by_node})

  # per-period percentage: flat, or the annual year-type ladder
  if flat:
    year_type = {}
    def pct_for(_key):
      return pct
  else:
    total_unimp = defaultdict(float)
    for (node, key), v in inflow_by_node.items():
      if node in rim_delta:
        total_unimp[key] += v
    if thresholds is None:
      thresholds = historical_thresholds()
    if thresholds is not None:
      lo, hi = thresholds
      year_type = {k: ('dry' if total_unimp[k] < lo else
                       'wet' if total_unimp[k] >= hi else 'middle')
                   for k in all_keys}
    else:
      if log:
        log.warning('env_flow: no historical unimpaired reference (%s); using '
                    'df-relative terciles (NOT fixed)' % DEFAULT_UNIMP_CSV)
      ranked = sorted(all_keys, key=lambda k: total_unimp[k])
      n = len(ranked)
      year_type = {}
      for rank, k in enumerate(ranked):
        frac = (rank + 0.5) / n
        year_type[k] = 'dry' if frac < 1/3 else ('middle' if frac < 2/3
                                                 else 'wet')
    def pct_for(key):
      return pct[year_type[key]]

  # per-tributary unimpaired inflow (rim inflows draining to the terminal
  # reservoir) and the reservoir-release reach arcs, keyed by period
  trib_unimp = defaultdict(float)   # (name, key) -> volume
  trib_arcs = defaultdict(list)     # (name, key) -> [(i,j,k)]
  for t in tribs:
    anc = _ancestors(rev, t['reservoir'])
    group = (rim_nodes & anc)
    if not group:
      raise ValueError('Tributary %s: no rim inflows drain to %s'
                       % (t['name'], t['reservoir']))
    for node in group:
      for key in all_keys:
        v = inflow_by_node.get((node, key), 0.0)
        if v:
          trib_unimp[(t['name'], key)] += v
    # reach arcs: reservoir -> reach
    reach_rows = df[(bi == t['reservoir']) & (bj == t['reach'])]
    if reach_rows.empty:
      raise ValueError('Tributary %s: no reach arc %s -> %s in network'
                       % (t['name'], t['reservoir'], t['reach']))
    for row in reach_rows.itertuples():
      key = keyfn(row.i)
      if key is not None:
        trib_arcs[(t['name'], key)].append((row.i, row.j, row.k))

  # existing (D-1641) required outflow per period, and aggregate outflow arcs
  is_req = (bi == DELTA_NODE) & (bj == REQ_TARGET)
  existing_req = defaultdict(float)
  for row in df[is_req].itertuples():
    key = keyfn(row.i)
    if key is not None:
      existing_req[key] += row.lower_bound
  is_out = (bi == DELTA_NODE) & bj.isin(OUTFLOW_TARGETS)
  agg_arcs = defaultdict(list)
  for row in df[is_out].itertuples():
    key = keyfn(row.i)
    if key is not None:
      agg_arcs[key].append((row.i, row.j, row.k))

  # assemble
  trib_out = []
  for t in tribs:
    for key in all_keys:
      arcs = trib_arcs.get((t['name'], key))
      if not arcs:
        continue
      rhs = pct_for(key) * trib_unimp.get((t['name'], key), 0.0)
      trib_out.append((t['name'], key, rhs, arcs))

  agg_out = []
  for key in all_keys:
    arcs = agg_arcs.get(key)
    if not arcs:
      continue
    trib_sum = sum(pct_for(key) * trib_unimp.get((t['name'], key), 0.0)
                   for t in tribs)
    rhs = max(existing_req.get(key, 0.0), trib_sum)
    agg_out.append((key, rhs, arcs))

  if log:
    log.info('env_flow: %d tributaries, %d %s (pct=%s)'
             % (len(tribs), len(all_keys),
                'months' if monthly else 'water years',
                pct if flat else 'year-type ladder'))
  return {'trib': trib_out, 'agg': agg_out, 'year_type': year_type}


def env_flow_rows(df, config, log=None, tribs=None):
  """Backend-neutral env-flow constraint specs.

  :returns: ``(req, specs)`` where ``req`` is the :func:`compute_requirements`
    dict and ``specs`` is a list of ``(coeffs, sense, rhs, label)`` ready for
    ``HighsNetworkModel.add_rows`` or a Pyomo ConstraintList. ``coeffs`` maps arc
    key ``(i, j, k)`` -> coefficient; ``sense`` is ``'>='``; ``label`` is
    ``('trib', name, wy)`` or ``('agg', wy)``. Aggregate rows are included only
    when ``config['aggregate']`` (default True).
  """
  tribs = tribs or load_tributaries(config.get('tributaries_csv'))
  if 'pct' in config:
    pct = config['pct']                    # flat float (drops the year-type ladder)
  else:
    pct = {'wet': config.get('pct_wet', DEFAULT_PCT['wet']),
           'middle': config.get('pct_middle', DEFAULT_PCT['middle']),
           'dry': config.get('pct_dry', DEFAULT_PCT['dry'])}

  req = compute_requirements(df, tribs, pct=pct,
                             thresholds=config.get('year_type_thresholds'),
                             monthly=config.get('monthly', False), log=log)

  def _coeffs(arcs):
    c = {}
    for a in arcs:  # accumulate so a repeated arc doubles, matching sum(X)
      c[a] = c.get(a, 0.0) + 1.0
    return c

  specs = [(_coeffs(arcs), '>=', rhs, ('trib', name, key))
           for (name, key, rhs, arcs) in req['trib']]
  if config.get('aggregate', True):
    specs += [(_coeffs(arcs), '>=', rhs, ('agg', key))
              for (key, rhs, arcs) in req['agg']]
  return req, specs


def add_env_flow_constraints(model, df, config, log=None, tribs=None):
  """Add per-tributary + aggregate env-flow constraints to a built model.

  Dispatches on the backend: a ``HighsNetworkModel`` (which exposes
  ``add_rows``) gets the row specs directly; a Pyomo model gets ConstraintLists
  built from ``model.X`` with the original 1-based label maps preserved.

  :param model: a built model — Pyomo (``model.X[i,j,k]`` defined) or a
    ``HighsNetworkModel``.
  :param df: the links DataFrame the model was built from
  :param config: dict — ``pct_wet``/``pct_middle``/``pct_dry`` (defaults
    0.55/0.40/0.30), ``aggregate`` (default True), ``tributaries_csv`` (optional),
    ``year_type_thresholds`` (optional (lo, hi); defaults to the fixed historical
    terciles from :func:`historical_thresholds`)
  """
  req, specs = env_flow_rows(df, config, log=log, tribs=tribs)
  trib_specs = [s for s in specs if s[3][0] == 'trib']
  agg_specs = [s for s in specs if s[3][0] == 'agg']

  if hasattr(model, 'add_rows'):        # HighsNetworkModel
    # optional penalized slack per row: a dry month pays a penalty instead of
    # going infeasible (feasibility backstop + Benders feasibility-cut source).
    penalty = config.get('relax_penalty')
    if penalty is None and config.get('relax'):
      penalty = 10.0 * float(df['cost'].abs().max())
    if penalty:
      col_specs, soft = [], []
      for (coeffs, sense, rhs, label) in specs:
        skey = ('env_slack',) + label
        col_specs.append((skey, 0.0, rhs, float(penalty)))   # slack in [0, rhs]
        c = dict(coeffs); c[skey] = 1.0
        soft.append((c, sense, rhs, label))
      model.add_columns(col_specs)       # slack columns first
      model.add_rows(soft)
      if log:
        log.info('env_flow: %d rows made soft (penalty %.3g/TAF)'
                 % (len(specs), float(penalty)))
    else:
      model.add_rows(specs)
  else:                                  # Pyomo model
    from pyomo.environ import ConstraintList
    # ConstraintList indices are 1-based in add order; keep parallel label maps
    model.env_trib = ConstraintList()
    model.env_trib_labels = {}
    for (coeffs, _s, rhs, label) in trib_specs:
      con = model.env_trib.add(
          sum(coef * model.X[a] for a, coef in coeffs.items()) >= rhs)
      model.env_trib_labels[con.index()] = (label[1], label[2])
    if agg_specs:
      model.env_agg = ConstraintList()
      model.env_agg_labels = {}
      for (coeffs, _s, rhs, label) in agg_specs:
        con = model.env_agg.add(
            sum(coef * model.X[a] for a, coef in coeffs.items()) >= rhs)
        model.env_agg_labels[con.index()] = label[1]

  if log:
    log.info('env_flow: added %d tributary + %d aggregate constraints'
             % (len(trib_specs), len(agg_specs)))
  return req
