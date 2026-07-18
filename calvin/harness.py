"""Fast lever-testing harness for CALVIN model design.

Build a small debug-enabled network once, then probe a model change — an
optional bounds/inflow transform and/or a constraint injector — in ~10 seconds
each, instead of a ~20-minute full 82-year solve.

Two solve modes, chosen automatically per probe:

* ``real``  — debug links stripped, real costs.  Gives Delta-outflow and
  shortage numbers.  Used when the (possibly constrained) network is feasible.
* ``debug`` — debug escape-valves kept, so the network is *always* solvable.
  The residual flow through the valves, localized by node, is a clean
  "how infeasible" signal for a lever that over-constrains.  This is the
  diagnostic the env-flow lever was missing: a bare short-window network is
  infeasible for spin-up reasons, but ``add_debug=True`` neutralizes that and
  isolates the infeasibility a lever actually introduces.

Extension protocols (each foundation lever fits one of these):

* constraint injector — ``fn(model, df, config, log=None) -> Any``,
  e.g. :func:`calvin.env_flow.add_env_flow_constraints`.
* transform — ``fn(df, config) -> df``, editing bounds/inflows before the
  model is built, e.g. a Colorado import cut or a GCM warm-shift.

Example
-------
    from calvin.harness import build_fixture, probe, compare
    from calvin.env_flow import add_env_flow_constraints

    df = build_fixture('../calvin-network-data/data')
    base  = probe(df, label='baseline')
    lever = probe(df, label='env-flow',
                  add_constraints=add_env_flow_constraints,
                  config={'aggregate': True})
    print(compare(base, lever))

The harness is for design iteration and feasibility, not final numbers — a
short window can't calibrate a lever against a full-record target.
"""
import os
import time
from dataclasses import dataclass, field

import pandas as pd
from pyomo.environ import value

from calvin.calvin import CALVIN, setup_logger
from calvin.network import load_network, build_matrix
from calvin.env_flow import DELTA_NODE, OUTFLOW_TARGETS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# my-models* is gitignored, so the cache and per-probe scratch stay out of git.
DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(BASE_DIR),
                                 'my-models', '.harness-cache')

# Residual valve flow (TAF, whole window) below this counts as feasible — a
# zero-debug feasible flow means the real (debug-stripped) network is feasible.
FEAS_TOL_TAF = 1.0


# ---------------------------------------------------------------------------
# Fixture: a small debug-enabled network, built once and cached
# ---------------------------------------------------------------------------
def build_fixture(data_path, start='1921-10', stop='1924-09',
                  constrain_ending='gw', add_debug=True,
                  node_lb_overrides=None, cache_path=None, rebuild=False,
                  log=None):
  """Build (or load from cache) a small debug-enabled links DataFrame.

  The default window is three water years (WY1922-24, which spans a dry year)
  so env-flow tercile year-typing is non-trivial, and ``constrain_ending='gw'``
  matches the study's fixed nogwod backdrop.  ``add_debug=True`` is what makes
  any short window solvable in debug mode.

  :param data_path: path to the calvin-network-data ``data`` directory.
  :param cache_path: where to cache the links CSV; defaults to a gitignored
    file under ``my-models/.harness-cache/`` keyed by the build parameters.
  :param rebuild: force a rebuild even if the cache exists.
  :returns: links DataFrame (columns i, j, k, cost, amplitude,
    lower_bound, upper_bound).
  """
  os.makedirs(DEFAULT_CACHE_DIR, exist_ok=True)
  log = log or setup_logger('harness', savedir=DEFAULT_CACHE_DIR)

  if cache_path is None:
    tag = '%s_%s_%s%s' % (start, stop, constrain_ending,
                          '_dbg' if add_debug else '')
    cache_path = os.path.join(DEFAULT_CACHE_DIR,
                              'fixture_%s.csv' % tag.replace(':', '-'))

  if os.path.isfile(cache_path) and not rebuild:
    log.info('harness: loading cached fixture %s' % cache_path)
    return pd.read_csv(cache_path)

  log.info('harness: building fixture %s..%s (constrain_ending=%s, debug=%s)'
           % (start, stop, constrain_ending, add_debug))
  network = load_network(data_path)
  df = build_matrix(network, start=start, stop=stop, add_debug=add_debug,
                    constrain_ending=constrain_ending,
                    node_lb_overrides=node_lb_overrides)
  df.to_csv(cache_path, index=False)
  log.info('harness: wrote %d links to %s' % (len(df), cache_path))
  return df


# ---------------------------------------------------------------------------
# Probe result
# ---------------------------------------------------------------------------
@dataclass
class ProbeResult:
  """Outcome of a single probe.  Metrics are per water year unless noted."""
  label: str
  mode: str                      # 'real' or 'debug'
  feasible: bool                 # real-cost feasible (True only in 'real' mode)
  n_water_years: int
  solve_seconds: float
  delta_outflow_taf: float | None = None   # TAF/yr out of D541 (both modes)
  exports_taf: float | None = None         # TAF/yr Banks+Tracy (both modes)
  shortage_cost_musd: float | None = None  # $M/yr  (real modes only)
  op_cost_musd: float | None = None        # $M/yr  (real modes only)
  debug_residual_taf: float = 0.0          # total valve flow (debug mode)
  debug_by_node: dict = field(default_factory=dict)  # node -> valve TAF
  relaxed_taf: float = 0.0                  # bound moved to reach feasibility

  def summary(self):
    head = ('[%s] mode=%s feasible=%s  (%d WY, %.1fs)'
            % (self.label or 'probe', self.mode, self.feasible,
               self.n_water_years, self.solve_seconds))
    lines = [head]
    if self.relaxed_taf:
      lines.append('  (relaxed %.1f TAF of bounds to reach feasibility)'
                   % self.relaxed_taf)
    if self.delta_outflow_taf is not None:
      lines.append('  Delta outflow = %.1f TAF/yr' % self.delta_outflow_taf)
    if self.exports_taf is not None:
      lines.append('  Banks+Tracy exports = %.1f TAF/yr' % self.exports_taf)
    if self.mode.startswith('real'):
      lines.append('  shortage cost = %.1f $M/yr | op cost = %.1f $M/yr'
                   % (self.shortage_cost_musd, self.op_cost_musd))
    else:
      top = ', '.join('%s=%.0f' % (k, v)
                      for k, v in list(self.debug_by_node.items())[:3])
      lines.append('  residual debug = %.1f TAF (top: %s)'
                   % (self.debug_residual_taf, top or 'none'))
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Probe: apply an optional transform + optional constraint injector, solve
# ---------------------------------------------------------------------------
def probe(df, *, transform=None, add_constraints=None, config=None,
          solver='highs', nproc=1, relax_to_feasible=False, relax_maxiter=8,
          logdir=None, label='probe', log=None, backend='pyomo'):
  """Solve a small network with an optional transform and/or constraint lever.

  Strategy: solve in debug mode first, because with the ``add_debug=True``
  valves kept the model is always solvable and the min-debug objective drives
  valve flow to zero if a feasible flow exists.  So residual below
  ``FEAS_TOL_TAF`` means the real (debug-stripped) network is feasible, and a
  larger residual localizes the infeasibility by node — showing *where* a lever
  over-constrains.  Only a feasible network gets a second real-cost solve for
  economics, which is guaranteed optimal and so never pays the infeasibility
  IIS cost.

  Debug mode gives feasibility but not economics (its objective is flattened, so
  flows aren't the economic allocation).  For an economic signal on a short
  window that has a small structural dead-end (e.g. a fixed-ending-storage
  reservoir the window can't drain), pass ``relax_to_feasible=True``: the debug
  loop relaxes just enough bound to clear the residual, then a real-cost solve
  runs on the relaxed network (``mode='real-relaxed'``), and ``relaxed_taf``
  reports how much bound was moved so you can judge the distortion.  Measure
  effects as differences between two same-window relaxed probes — the relaxation
  and end-of-horizon effects largely cancel.

  :param df: a links DataFrame from :func:`build_fixture` (must carry the
    ``add_debug=True`` valves).
  :param transform: optional ``fn(df, config) -> df`` applied before build.
  :param add_constraints: optional constraint injector ``fn(model, df, config,
    log=None)``, or a list of them to STACK (e.g. the environmental scenario =
    percent-of-unimpaired outflow + Delta export cap). Called after
    ``create_pyomo_model`` (mirrors ``CALVINCap``'s order).
  :param config: dict passed through to ``transform`` and ``add_constraints``.
  :param relax_to_feasible: if the network is infeasible, relax minimal bounds
    to feasibility and solve real-cost anyway (for an economic signal).
  :param backend: ``'pyomo'`` (default) or ``'highs'`` — which model backend to
    build and solve through. Metrics are backend-neutral (they read
    ``model_to_dataframe``), so the two backends are interchangeable here.
  :returns: a :class:`ProbeResult`.
  """
  config = config or {}
  logdir = logdir or DEFAULT_CACHE_DIR
  os.makedirs(logdir, exist_ok=True)
  log = log or setup_logger('harness', savedir=logdir)

  work = df.copy()
  if transform is not None:
    work = transform(work, config)
  n_wy = _n_water_years(work)

  # CALVIN reads a links CSV; write the (transformed) network to a scratch file.
  safe = ''.join(c if c.isalnum() else '_' for c in label) or 'probe'
  csv_path = os.path.join(logdir, 'harness_links_%s.csv' % safe)
  work.to_csv(csv_path, index=False)

  # add_constraints may be a single injector or a list to STACK (e.g. the
  # environmental scenario = percent-of-unimpaired outflow + Delta export cap).
  injectors = ([] if add_constraints is None else
               list(add_constraints) if isinstance(add_constraints, (list, tuple))
               else [add_constraints])

  def _build(cal, debug_mode):
    if backend == 'highs':
      target = cal.create_highs_model(debug_mode=debug_mode)
    else:
      cal.create_pyomo_model(debug_mode=debug_mode)
      target = cal.model
    for inject in injectors:
      inject(target, cal.df, config, log=log)
    return cal

  def _solve(cal, debug_mode, maxiter=10):
    if backend == 'highs':
      return cal.solve_highs_model(solver=solver, nproc=nproc,
                                   debug_mode=debug_mode, maxiter=maxiter)
    return cal.solve_pyomo_model(solver=solver, nproc=nproc,
                                 debug_mode=debug_mode, maxiter=maxiter)

  def _new_calvin():
    return CALVIN(csv_path, log_name='harness_%s' % safe, logdir=logdir)

  def _real_result(cal, mode, seconds, relaxed_taf=0.0):
    mdf = cal.model_to_dataframe()
    outflow, short_c, op_c = _real_metrics(mdf, n_wy)
    return ProbeResult(label=label, mode=mode, feasible=True,
                       n_water_years=n_wy, solve_seconds=seconds,
                       delta_outflow_taf=outflow, exports_taf=_exports(mdf, n_wy),
                       shortage_cost_musd=short_c, op_cost_musd=op_c,
                       relaxed_taf=relaxed_taf)

  # --- debug solve first: always feasible; residual is the feasibility oracle.
  # maxiter=1 reads the residual as-is, before the fix_debug_flows bound-
  # relaxation heuristic mutates anything.
  t0 = time.perf_counter()
  cal = _build(_new_calvin(), debug_mode=True)
  _solve(cal, debug_mode=True, maxiter=1)
  residual, by_node = _debug_residual(cal)
  dt = time.perf_counter() - t0

  if residual <= FEAS_TOL_TAF:
    # feasible: real-cost solve for economics (guaranteed optimal, no IIS)
    t1 = time.perf_counter()
    cal = _build(_new_calvin(), debug_mode=False)
    _solve(cal, debug_mode=False)
    return _real_result(cal, 'real', dt + time.perf_counter() - t1)

  if not relax_to_feasible:
    log.info('harness [%s]: infeasible under real costs (residual %.1f TAF); '
             'reporting debug signal' % (label, residual))
    mdf = cal.model_to_dataframe()
    return ProbeResult(label=label, mode='debug', feasible=False,
                       n_water_years=n_wy, solve_seconds=dt,
                       delta_outflow_taf=_delta_outflow(mdf, n_wy),
                       exports_taf=_exports(mdf, n_wy),
                       debug_residual_taf=residual, debug_by_node=by_node)

  # --- relax to feasibility, then real-cost solve for an economic signal ---
  t1 = time.perf_counter()
  # continue the debug loop from the maxiter=1 state; fix_debug_flows relaxes
  # minimal real bounds until the residual clears.
  converged = _solve(cal, debug_mode=True, maxiter=relax_maxiter)
  adj = cal.get_bound_adjustments()
  relaxed_taf = float(adj[['lb_delta', 'ub_delta']].abs().sum().sum()) if len(adj) else 0.0
  if not converged:
    resid2, by_node2 = _debug_residual(cal)
    log.warning('harness [%s]: could not relax to feasibility '
                '(%.1f TAF residual remains)' % (label, resid2))
    mdf = cal.model_to_dataframe()
    return ProbeResult(label=label, mode='debug', feasible=False,
                       n_water_years=n_wy, solve_seconds=dt + time.perf_counter() - t1,
                       delta_outflow_taf=_delta_outflow(mdf, n_wy),
                       exports_taf=_exports(mdf, n_wy),
                       debug_residual_taf=resid2, debug_by_node=by_node2)
  # rebuild real-cost on the now-relaxed bounds (the build strips DBUG and reads
  # the mutated cal.df) and re-inject the constraints
  _build(cal, debug_mode=False)
  _solve(cal, debug_mode=False)
  log.info('harness [%s]: relaxed %.1f TAF of bounds to reach feasibility'
           % (label, relaxed_taf))
  return _real_result(cal, 'real-relaxed', dt + time.perf_counter() - t1,
                      relaxed_taf=relaxed_taf)


def compare(baseline, lever):
  """Pretty diff of a lever probe against a baseline probe.

  When either probe fell back to debug mode, the key signal is the *added*
  residual the lever introduces over the baseline, localized by node — a
  feasible lever adds ~0, an over-constraining one piles residual onto exactly
  the nodes it touches (e.g. tributary reaches for env-flow).
  """
  lines = ['%s vs %s: lever mode=%s feasible=%s'
           % (lever.label, baseline.label, lever.mode, lever.feasible)]
  if 'debug' in (lever.mode, baseline.mode):
    b, l = baseline.debug_residual_taf, lever.debug_residual_taf
    lines.append('  residual valve flow: %.1f -> %.1f TAF (Δ %+.1f) '
                 '[lower = more feasible]' % (b, l, l - b))
    added = {n: v - baseline.debug_by_node.get(n, 0.0)
             for n, v in lever.debug_by_node.items()
             if v - baseline.debug_by_node.get(n, 0.0) > 1.0}
    if added:
      lines.append('  added residual by node (lever over baseline):')
      for node, d in sorted(added.items(), key=lambda kv: -kv[1])[:10]:
        lines.append('    %-26s %+8.1f TAF' % (node, d))
    else:
      lines.append('  no material added residual — lever is feasible on the '
                   'real system')
  _diff_line(lines, 'Delta outflow', baseline.delta_outflow_taf,
             lever.delta_outflow_taf, 'TAF/yr')
  _diff_line(lines, 'Banks+Tracy exports', baseline.exports_taf,
             lever.exports_taf, 'TAF/yr')
  _diff_line(lines, 'Shortage cost', baseline.shortage_cost_musd,
             lever.shortage_cost_musd, '$M/yr')
  return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Metric helpers (cheap; no full postprocess())
# ---------------------------------------------------------------------------
def _diff_line(lines, name, base, lev, unit):
  if base is not None and lev is not None:
    lines.append('  %s: %.1f -> %.1f %s (Δ %+.1f)'
                 % (name, base, lev, unit, lev - base))


def _n_water_years(df):
  """Count distinct water years among timestamped node ids."""
  wys = set()
  for node in df['j']:
    parts = str(node).split('.', 1)
    if len(parts) == 2:
      d = parts[1]
      y, mo = int(d[:4]), int(d[5:7])
      wys.add(y + 1 if mo >= 10 else y)
  return max(len(wys), 1)


def _delta_outflow(mdf, n_wy):
  """TAF/yr leaving D541 via Req_Delta + Surp_Delta.

  Req_Delta/Surp_Delta are timestamped nodes (e.g. ``Req_Delta.1921-10-31``),
  so match on the base name, not the bare boundary string.
  """
  i_base = mdf.i.str.split('.').str[0]
  j_base = mdf.j.str.split('.').str[0]
  m = mdf[i_base.eq(DELTA_NODE) & j_base.isin(OUTFLOW_TARGETS)]
  return m.flow.sum() / n_wy


def _exports(mdf, n_wy):
  """TAF/yr of Delta exports = Banks (PMP_Banks→D800) + Tracy (PMP_Tracy→D701)."""
  i_base = mdf.i.str.split('.').str[0]
  j_base = mdf.j.str.split('.').str[0]
  banks = i_base.eq('PMP_Banks') & j_base.eq('D800')
  tracy = i_base.eq('PMP_Tracy') & j_base.eq('D701')
  return mdf[banks | tracy].flow.sum() / n_wy


def _real_metrics(mdf, n_wy):
  """Delta outflow, shortage cost, and op cost from a real-cost solve.

  Mirrors ``compute_network_costs`` in scripts/calvin-pf-cap.py: drop storage
  carryover (SR/GW -> FINAL) and reservoir spill (SR -> SR), treat bounded
  negative-cost arcs as demand (shortage = unmet upper bound), and price it.
  Only the shortage *cost* is reported — the raw unmet volume is dominated by
  high-bound low-penalty arcs and isn't a clean shortage measure; use exports /
  Delta outflow for water quantities.
  """
  outflow = _delta_outflow(mdf, n_wy)
  cost_links = mdf.drop(mdf[((mdf.i.str.contains('SR')) |
                             (mdf.i.str.contains('GW'))) &
                            (mdf.j.str.contains('FINAL'))].index)
  cost_links = cost_links.loc[~cost_links.index.str.contains('DBUG')]
  cost_links = cost_links.drop(cost_links[(cost_links.i.str.contains('SR')) &
                                          (cost_links.j.str.contains('SR'))].index)
  short_links = cost_links.loc[cost_links.cost < 0]
  short_links = short_links.loc[short_links.upper_bound < 1e6]
  unmet = short_links.upper_bound - short_links.flow
  short_cost = -1 * (unmet * short_links.cost).sum() / n_wy / 1e3
  op_links = cost_links.loc[cost_links.cost > 0]
  op_cost = (op_links.flow * op_links.cost).sum() / n_wy / 1e3
  return outflow, short_cost, op_cost


def _debug_residual(cal):
  """Total and per-node residual flow through the DBUG escape valves.

  Matches the ``fix_debug_flows`` selection: DBUGSRC->node injections (deficit)
  and node->DBUGSNK drains (surplus); the SOURCE->DBUGSRC / DBUGSNK->SINK
  connectors are excluded.  A lever that over-constrains shows up as extra
  injection on exactly the nodes it touches.
  """
  if getattr(cal, '_backend', 'pyomo') == 'highs':
    flows = cal.hmodel.flows()
    get = lambda i, j, k: flows.get((i, j, k), 0.0)
  else:
    m = cal.model
    get = lambda i, j, k: value(m.X[i, j, k]) or 0.0
  by_node = {}
  total = 0.0
  for (i, j, k) in cal.links:
    if i == 'DBUGSRC':
      node = j
    elif j == 'DBUGSNK':
      node = i
    else:
      continue
    v = get(i, j, k) or 0.0
    if v <= 1e-6:
      continue
    total += v
    by_node[node] = by_node.get(node, 0.0) + v
  by_node = dict(sorted(by_node.items(), key=lambda kv: -kv[1]))
  return total, by_node
