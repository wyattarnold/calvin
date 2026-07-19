import os
import sys
import logging
import datetime

from pyomo.environ import *
from pyomo.opt import TerminationCondition
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def setup_logger(log_name, savedir=None, console_level=logging.INFO):
  """
  Create the logger

  :param console_level: (int) logging level for the console handler (default INFO).
    Pass ``logging.WARNING`` in EA worker processes to suppress per-solve console
    noise while still writing full DEBUG output to the log file.
  """
  if savedir is not None:
    log_name = os.path.join(savedir,log_name)
  logger = logging.getLogger(log_name)
  if not logger.hasHandlers():  # hasHandlers will only be True if someone already called CALVIN with the same log_name in the same session
    logger.setLevel("DEBUG")
    screen_handler = logging.StreamHandler(sys.stdout)
    screen_handler.setLevel(console_level)
    screen_formatter = logging.Formatter('%(levelname)s - %(message)s')
    screen_handler.setFormatter(screen_formatter)
    logger.addHandler(screen_handler)

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    file_handler = logging.FileHandler("{}.{}.log".format(log_name, timestamp))
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
  return logger


class CALVIN():

  def __init__(self, linksfile, ic=None, log_name="calvin", logdir=None):
    """
    Initialize CALVIN model object.

    :param linksfile: (string) CSV file containing network link information
    :param ic: (dict) Initial storage conditions for surface reservoirs
                only used for annual optimization
    :param log_name: (string) Logger name; used as the log filename stem.
                Defaults to "calvin" → writes calvin.log in logdir (or cwd).
    :param logdir: (string) Directory to write the log file. Defaults to the
                directory containing linksfile.
    :returns: CALVIN model object
    """

    # Default logdir to the directory containing the linksfile
    if logdir is None:
      logdir = os.path.dirname(os.path.abspath(linksfile))
    if not os.path.isdir(logdir):
      os.makedirs(logdir)
    self.log = setup_logger(log_name, savedir=logdir)

    df = pd.read_csv(linksfile)
    df['link'] = df.i.map(str) + '_' + df.j.map(str) + '_' + df.k.map(str)
    df.set_index('link', inplace=True)

    self.df = df
    self.linksfile = os.path.splitext(linksfile)[0] # filename w/o extension

    SR_stats = pd.read_csv(os.path.join(BASE_DIR, 'data', 'SR_stats.csv'), index_col=0).to_dict()
    self.min_storage = SR_stats['min']
    self.max_storage = SR_stats['max']

    if ic:
      self.apply_ic(ic)

    # a few network fixes to make things work
    self.add_ag_region_sinks()

    self.nodes = pd.unique(self.df[['i','j']].values.ravel()).tolist()
    self.links = list(zip(self.df.i, self.df.j, self.df.k))
    self.networkcheck() # make sure things aren't broken

    # Snapshot bounds after all initialization; used by get_bound_adjustments()
    self._initial_bounds = self.df[['lower_bound', 'upper_bound']].copy()

  def apply_ic(self, ic):
    """
    Set initial storage conditions.

    :param ic: (dict) initial storage values
    :returns: nothing, but modifies the model object
    """
    for k in ic:
      ix = (self.df.i.str.contains('INITIAL') &
            self.df.j.str.contains(k))
      self.df.loc[ix, ['lower_bound','upper_bound']] = ic[k]

  def inflow_multiplier(self, x):
    """
    Multiply all network inflows by a constant.

    :param x: (float) value to multiply inflows
    :returns: nothing, but modifies the model object
    """
    ix = self.df.i.str.contains('INFLOW')
    self.df.loc[ix, ['lower_bound','upper_bound']] *= x

  def eop_constraint_multiplier(self, x):
    """
    Set end-of-period storage constraints as a fraction of maximum 
    available storage. Needed for limited foresight (annual) optimization.

    :param x: (float) fraction of maximum storage to set lower bound
    :returns: nothing, but modifies the model object
    """
    for k in self.max_storage:
      ix = (self.df.i.str.contains(k) &
            self.df.j.str.contains('FINAL'))
      lb = self.min_storage[k] + (self.max_storage[k]-self.min_storage[k])*x
      self.df.loc[ix,'lower_bound'] = lb
      self.df.loc[ix,'upper_bound'] = self.max_storage[k]

  def no_gw_overdraft(self):
    """
    Impose constraints to prevent groundwater overdraft

    :returns: nothing, but modifies the model object
    """
    ix_i = (self.df.i.str.contains('INITIAL') & self.df.j.str.contains('GW')) # initial groundwater storage
    ix_f = (self.df.i.str.contains('GW') & self.df.j.str.contains('FINAL')) # final groundwater storage
    
    initial = self.df.loc[ix_i, ['lower_bound','upper_bound']].values
    final = self.df.loc[ix_f, ['lower_bound','upper_bound']].values
    mask = final < initial # element-wise comparison, masking because for some gw basins final > initial (negative overdraft)
    final[mask] = initial[mask] # update only where condition holds, set final storage as initial storage (no overdraft)
    self.df.loc[ix_f, ['lower_bound','upper_bound']] = final

  def networkcheck(self):
    """
    Confirm constraint feasibility for the model object.
    (No inputs or outputs)
    :raises: ValueError when infeasibilities are identified.
    """
    nodes = self.nodes
    links = self.df.values

    num_in = {n: 0 for n in nodes}
    num_out = {n: 0 for n in nodes}
    lb_in = {n: 0 for n in nodes} 
    lb_out = {n: 0 for n in nodes}
    ub_in = {n: 0 for n in nodes} 
    ub_out = {n: 0 for n in nodes}

    # loop over links
    for l in links:
      lb = float(l[5])
      ub = float(l[6])
      num_in[l[1]] += 1
      lb_in[l[1]] += lb
      ub_in[l[1]] += ub
      num_out[l[0]] += 1
      lb_out[l[0]] += lb
      ub_out[l[0]] += ub

      if lb > ub:
        raise ValueError('lb > ub for link %s' % (l[0]+'-'+l[1]))
    
    for n in nodes:
      if num_in[n] == 0 and n not in ['SOURCE','SINK']:
        raise ValueError('no incoming link for ' + n)
      if num_out[n] == 0 and n not in ['SOURCE','SINK']:
        raise ValueError('no outgoing link for ' + n)

      if ub_in[n] < lb_out[n]:
        raise ValueError('ub_in < lb_out for %s (%d < %d)' % (n, ub_in[n], lb_out[n]))
      if lb_in[n] > ub_out[n]:
        raise ValueError('lb_in > ub_out for %s (%d > %d)' % (n, lb_in[n], ub_out[n]))

  def add_ag_region_sinks(self):
    """
    Hack to get rid of surplus water at no cost from agricultural regions.
    Called internally when model is initialized.

    :returns: nothing, but modifies the model object
    """
    df = self.df
    links = df[df.i.str.contains('HSU') & ~df.j.str.contains('DBUG')].copy(deep=True)
    if not links.empty:
      maxub = links.upper_bound.max()
      links.j = links.apply(lambda l: 'SINK.'+l.i.split('.')[1], axis=1)
      links.cost = 0.0
      links.amplitude = 1.0
      links.lower_bound = 0.0
      links.upper_bound = maxub
      links['link'] = links.i.map(str) + '_' + links.j.map(str) + '_' + links.k.map(str)
      links.set_index('link', inplace=True)
      self.df = pd.concat([self.df, links.drop_duplicates()])


  def get_bound_adjustments(self):
    """
    Return a dataframe of links whose bounds were modified by fix_debug_flows,
    showing the initial and final lower/upper bounds and the net delta.

    :returns: DataFrame with columns i, j, k, lb_init, lb_final, lb_delta,
              ub_init, ub_final, ub_delta — only rows where at least one bound changed.
    """
    cur = self.df[['i', 'j', 'k', 'lower_bound', 'upper_bound']].copy()
    cur = cur.join(self._initial_bounds.rename(
        columns={'lower_bound': 'lb_init', 'upper_bound': 'ub_init'}))
    cur['lb_final'] = cur['lower_bound']
    cur['ub_final'] = cur['upper_bound']
    cur['lb_delta'] = cur['lb_final'] - cur['lb_init']
    cur['ub_delta'] = cur['ub_final'] - cur['ub_init']
    changed = cur[(cur['lb_delta'] != 0) | (cur['ub_delta'] != 0)]
    return changed[['i', 'j', 'k', 'lb_init', 'lb_final', 'lb_delta',
                    'ub_init', 'ub_final', 'ub_delta']].copy()

  def remove_debug_links(self):
    """
    Remove debug links from model object.

    :returns: dataframe of links, excluding debug links.
    """
    df = self.df
    ix = df.index[df.index.str.contains('DBUG')]
    df.drop(ix, inplace=True, axis=0)
    self.nodes = pd.unique(df[['i','j']].values.ravel()).tolist()
    self.links = list(zip(df.i,df.j,df.k))
    return df


  def create_pyomo_model(self, debug_mode=False, debug_cost=2e10, cosvf_mode=False,
                         save_final_csv=False):
    """
    Use link data to create Pyomo model (constraints and objective function)
    But do not solve yet.

    :param debug_mode: (boolean) Whether to run in debug mode.
      Use when there may be infeasibilities in the network.
    :param debug_cost: When in debug mode, assign this cost ($/AF) to flow on debug links.
      This should be an arbitrarily high number.
    :param cosvf_mode: (boolean) When in COSVF mode, use debug links but preserve all network link costs.
    :param save_final_csv: (boolean) Whether to save final network as csv after all modifications.
    :returns: nothing, but creates the model object (self.model)
    """

    # work on a local copy of the dataframe
    if not debug_mode and self.df.index.str.contains('DBUG').any():
      # previously ran in debug mode, but now done
      df = self.remove_debug_links()
    else:
      df = self.df

    if save_final_csv:
        df.to_csv(self.linksfile + '-final.csv')

    if not cosvf_mode: self.log.info('Creating Pyomo Model (debug=%s)' % debug_mode)

    model = ConcreteModel()

    model.N = Set(initialize=self.nodes)
    model.k = Set(initialize=range(15))
    model.A = Set(within=model.N*model.N*model.k,
                  initialize=self.links, ordered=True)
    model.source = Param(initialize='SOURCE', within=Any)
    model.sink = Param(initialize='SINK', within=Any)

    # Pre-build parameter dicts for fast O(1) lookup (avoids repeated string concat + df.loc)
    link_set = set(self.links)
    ub_dict = {}
    lb_dict = {}
    amp_dict = {}
    cost_dict = {}
    for row in df.itertuples():
      key = (row.i, row.j, row.k)
      if key not in link_set:
        continue
      ub_dict[key] = row.upper_bound
      lb_dict[key] = row.lower_bound
      amp_dict[key] = row.amplitude
      if debug_mode and ('DBUG' in str(row.i) + '_' + str(row.j)):
        cost_dict[key] = debug_cost
      elif debug_mode and not cosvf_mode:
        cost_dict[key] = 1.0
      else:
        cost_dict[key] = row.cost

    model.u = Param(model.A, initialize=ub_dict, mutable=True)
    model.l = Param(model.A, initialize=lb_dict, mutable=True)
    model.a = Param(model.A, initialize=amp_dict)
    model.c = Param(model.A, initialize=cost_dict, mutable=True)

    # The flow over each arc
    model.X = Var(model.A, within=Reals)

    # Minimize total cost
    def obj_fxn(model):
      return sum(model.c[i,j,k]*model.X[i,j,k] for (i,j,k) in model.A)
    model.total = Objective(rule=obj_fxn, sense=minimize)

    # Enforce an upper bound limit on the flow across each arc
    def limit_rule_upper(model, i, j, k):
      return model.X[i,j,k] <= model.u[i,j,k]
    model.limit_upper = Constraint(model.A, rule=limit_rule_upper)

    # Enforce a lower bound limit on the flow across each arc
    def limit_rule_lower(model, i, j, k):
      return model.X[i,j,k] >= model.l[i,j,k]
    model.limit_lower = Constraint(model.A, rule=limit_rule_lower)

    # Build arc_in/arc_out dicts in pure Python (no Pyomo overhead)
    arcs_in = {}
    arcs_out = {}
    for i, j, k in self.links:
      arcs_in.setdefault(j, []).append((i, j, k))
      arcs_out.setdefault(i, []).append((i, j, k))

    # Enforce flow through each node (mass balance)
    def flow_rule(model, node):
      if node in ('SOURCE', 'SINK'):
          return Constraint.Skip
      outflow  = sum(model.X[i,j,k]/model.a[i,j,k] for i,j,k in arcs_out[node])
      inflow = sum(model.X[i,j,k] for i,j,k in arcs_in[node])
      return inflow == outflow
    model.flow = Constraint(model.N, rule=flow_rule)

    model.dual = Suffix(direction=Suffix.IMPORT)

    self.model = model
    self._backend = 'pyomo'


  def solve_pyomo_model(self, solver='highs', nproc=1, debug_mode=False, maxiter=10, tee=False, save_json=False,
                        solver_options=None):
    """
    Solve Pyomo model (must be called after create_pyomo_model)

    :param solver: (string) solver name. glpk, cplex, cbc, gurobi.
    :param nproc: (int) number of processors. 1=serial.
    :param debug_mode: (boolean) Whether to run in debug mode.
      Use when there may be infeasibilities in the network.
    :param maxiter: (int) maximum iterations for debug mode.
    :returns: nothing, but assigns results to self.model.solutions.
    :raises: RuntimeError, if problem is found to be infeasible.
    :param tee: (boolean) Whether to show solver progress in the console
    :param save_json: (boolean) Whether to save raw solver outputs as a json
    :param solver_options: (dict) extra options passed through to the solver,
      e.g. {'solver': 'ipm', 'run_crossover': 'on'} to run HiGHS barrier with
      crossover on a one-shot PF solve. Keep crossover on when duals matter.
    """

    from pyomo.opt import SolverFactory
    opt = SolverFactory(solver)

    if nproc > 1 and solver != 'glpk':
      opt.options['threads'] = nproc

    if solver == 'cplex':
      opt.options['lpmethod'] = 4  # Barrier without crossover

    if solver_options:
      opt.options.update(solver_options)
    
    if debug_mode:
      run_again = True
      i = 0
      vol_total = 0
      prev_debug = float('inf')
      stalled = False

      while run_again and i < maxiter:
        self.log.info('-----Solving Pyomo Model (debug=%s)' % debug_mode)
        self.results = opt.solve(self.model)
        self.log.info('Finished. Fixing debug flows...')
        run_again, vol, total_debug = self.fix_debug_flows()
        i += 1
        vol_total += vol

        if run_again and total_debug >= prev_debug * 0.99:
          stalled = True
          break
        prev_debug = total_debug

      if not run_again:
        self.log.info('All debug flows eliminated (iter=%d, vol=%0.2f)' % (i, vol_total))
        return True
      elif stalled:
        self.log.warning('Debug stalled at %.2e TAF after %d iterations; '
                         'no further improvement possible.' % (total_debug, i))
        return False
      else:
        self.log.warning('Debug mode maximum iterations reached (%d); '
                         '%.2e TAF of debug flows remain.' % (maxiter, total_debug))
        return False

    else:
      self.log.info('-----Solving Pyomo Model (debug=%s)' % debug_mode)
      self.results = opt.solve(self.model, tee=tee, load_solutions=False)

      if self.results.solver.termination_condition == TerminationCondition.optimal:
        self.log.info('Optimal Solution Found (debug=%s).' % debug_mode)
        self.model.solutions.load_from(self.results)
        if save_json:
            self.model.solutions.store_to(self.results)
            self.results.write(filename='results.json', format='json')
        return True
      else:
        self.log.error('Solver status: %s' % self.results.solver.status)
        self.log.error('Termination condition: %s' % self.results.solver.termination_condition)
        if self.results.solution:
          self.log.error('Solution status: %s' % self.results.solution.status)
        self._log_iis()
        raise RuntimeError('Problem Infeasible. Run again starting from debug mode.')


  def _log_iis(self):
    """
    Write the current Pyomo model to a temp LP file, re-solve with HiGHS directly,
    and log the Irreducible Infeasible Subsystem (IIS) — the minimal set of conflicting
    constraints — to help diagnose the root cause of infeasibility.

    Requires ``highspy`` to be installed. Silently skips if not available.
    """
    try:
      import highspy
    except ImportError:
      self.log.warning('highspy not available; skipping IIS analysis.')
      return
    import tempfile

    # Write the Pyomo model to a temp LP with symbolic names so HiGHS
    # column/row names map back to our network link/node names.
    tmp = tempfile.NamedTemporaryFile(suffix='.lp', delete=False)
    tmp.close()
    try:
      self.model.write(tmp.name, io_options={'symbolic_solver_labels': True})

      h = highspy.Highs()
      h.setOptionValue('output_flag', False)
      # Use Irreducible strategy: FromRay (default) fails when bounds are
      # expressed as constraint rows rather than column bounds (Pyomo's format).
      h.setOptionValue('iis_strategy',
                       int(highspy._core.kIisStrategyIrreducible))
      h.readModel(tmp.name)
      h.run()

      if h.getModelStatus() != highspy.HighsModelStatus.kInfeasible:
        self.log.warning('IIS solver did not confirm infeasibility; skipping IIS.')
        return

      iis_status, iis = h.getIis()
      if iis_status != highspy.HighsStatus.kOk or not iis.valid_:
        self.log.warning('IIS computation failed or returned invalid result.')
        return

      UPPER = int(highspy._core.kIisBoundStatusUpper)

      lp = h.getLp()
      col_names = lp.col_names_
      row_names = lp.row_names_

      self.log.error('--- IIS: %d conflicting variable bounds, %d conflicting constraints ---'
                     % (len(iis.col_index_), len(iis.row_index_)))

      def _log_iis_items(indices, bounds, names, label):
        for idx, bound_status in zip(indices, bounds):
          bound = 'UB' if bound_status == UPPER else 'LB'
          name  = names[idx] if idx < len(names) else str(idx)
          self.log.error('  %s %s (%s)' % (label, name, bound))

      _log_iis_items(iis.col_index_, iis.col_bound_, col_names, 'Variable')
      _log_iis_items(iis.row_index_, iis.row_bound_, row_names, 'Constraint')

    finally:
      os.unlink(tmp.name)


  def _raise_upper_bounds(self, dbl, df, model, flow_val):
    """Raise UB on all outgoing links from a DBUGSNK node to relieve surplus water."""
    vol = 0
    raiselinks = df[(df.i == dbl[0]) & ~df.j.str.contains('DBUGSNK')].values
    for l in raiselinks:
      s2 = tuple(l[0:3])
      iv = model.u[s2].value
      v = flow_val * 1.2
      model.u[s2].value += v
      vol += v
      self.log.info('%s UB raised by %0.2f (%0.2f%%)' % (l[0] + '_' + l[1], v, v * 100 / iv))
      df.loc['_'.join(str(x) for x in l[0:3]), 'upper_bound'] = model.u[s2].value
    return vol

  def _lower_downstream_bounds(self, dbl, df, model, flow_val, max_depth=10):
    """Reduce LB on downstream links of a DBUGSRC node to relieve deficit water."""
    vol_to_reduce = max(flow_val * 1.2, 0.5)
    self.log.info('Volume to reduce: %.2e' % vol_to_reduce)

    children = [dbl[1]]
    for _ in range(max_depth):
      children += df[df.i.isin(children) & ~df.j.str.contains('DBUGSNK')].j.tolist()
    children = set(children)

    reducelinks = (df[df.i.isin(children) & (df.lower_bound > 0)]
                   .sort_values(by='lower_bound', ascending=False).values)
    if reducelinks.size == 0:
      raise RuntimeError('Not possible to reduce LB on links with origin %s by volume %0.2f'
                         % (dbl[1], vol_to_reduce))

    carryover = ['SR_', 'INITIAL', 'FINAL', 'GW_']
    vol = 0
    for l in reducelinks:
      if vol_to_reduce == 0:
        break
      s2 = tuple(l[0:3])
      iv = model.l[s2].value
      dl = model.dual[model.limit_lower[s2]] if s2 in model.limit_lower else 0.0
      if iv > 0 and dl > 1e6:
        v = min(vol_to_reduce, iv)
        if any(c in l[0] for c in carryover) and any(c in l[1] for c in carryover):
          v = min(v, max(25.0, 0.1 * iv))
        model.l[s2].value -= v
        vol_to_reduce -= v
        vol += v
        self.log.info('%s LB reduced by %.2e (%0.2f%%). Dual=%.2e'
                      % (l[0] + '_' + l[1], v, v * 100 / iv, dl))
        df.loc['_'.join(str(x) for x in l[0:3]), 'lower_bound'] = model.l[s2].value

    if vol_to_reduce > 0:
      self.log.info('Debug -> %s: could not reduce full amount (%.2e left)' % (dbl[1], vol_to_reduce))
    return vol

  def fix_debug_flows(self, tol=1e-7):
    """
    Find infeasible constraints where debug flows occur.
    Fix them by either raising the UB (DBUGSNK) or lowering the LB (DBUGSRC).

    :param tol: (float) Tolerance to identify nonzero debug flows
    :returns run_again: (boolean) whether debug mode needs to run again
    :returns vol: (float) total volume of constraint changes
    :returns total_debug: (float) total debug flow volume in this iteration
      also modifies the model object.
    """
    df, model = self.df, self.model
    dbix = (df.i.str.contains('DBUGSRC') | df.j.str.contains('DBUGSNK'))
    debuglinks = df[dbix].values

    total_debug = sum(max(model.X[tuple(dbl[0:3])].value or 0, 0) for dbl in debuglinks)

    run_again = False
    vol_total = 0

    for dbl in debuglinks:
      s = tuple(dbl[0:3])
      flow_val = model.X[s].value
      if flow_val <= tol:
        continue

      run_again = True
      if 'DBUGSNK' in dbl[1]:
        vol_total += self._raise_upper_bounds(dbl, df, model, flow_val)
      elif 'DBUGSRC' in dbl[0]:
        vol_total += self._lower_downstream_bounds(dbl, df, model, flow_val)

    self.df, self.model = df, model
    return run_again, vol_total, total_debug

  def model_to_dataframe(self):
      """
      Converts the model to a pandas dataframe.
      Useful for computing objective values (costs) without having to postprocess.

      :returns model_df: (Pandas dataframe) Dataframe of upper_bound, cost,
        and flow (solution) values for each link
      """
      if getattr(self, '_backend', 'pyomo') == 'highs':
        return self.hmodel.to_dataframe()

      def _param_series(param, col):
        keys = list(param.keys())
        return pd.DataFrame(
          [(i, j, k, param[i, j, k].value) for i, j, k in keys],
          columns=['i', 'j', 'k', col]
        ).set_index(['i', 'j', 'k'])

      m = self.model
      model_df = (
        _param_series(m.X, 'flow')
        .join(_param_series(m.c, 'cost'), how='inner')
        .join(_param_series(m.l, 'lower_bound'))
        .join(_param_series(m.u, 'upper_bound'))
        .reset_index()
      )
      model_df['link'] = model_df.i.map(str) + '-' + model_df.j.map(str) + '-' + model_df.k.map(str)
      model_df.set_index('link', inplace=True)

      return model_df

  # -------------------------------------------------------------------------
  # Direct-HiGHS backend (alongside Pyomo).  Builds the same network LP into a
  # persistent highspy.Highs() model, skipping the Pyomo object graph and the
  # LP-file handoff.  See calvin/highs_model.py.  The result surface
  # (model_to_dataframe, postprocess) is backend-neutral; COSVF stays on Pyomo.
  # -------------------------------------------------------------------------
  def create_highs_model(self, debug_mode=False, debug_cost=2e10, cosvf_mode=False):
    """Build the network LP directly in HiGHS.  Mirrors create_pyomo_model's df
    preparation and the three cost regimes, then assembles a HighsNetworkModel.

    :returns: the HighsNetworkModel (also stored as ``self.hmodel``).
    """
    from calvin.highs_model import HighsNetworkModel

    if not debug_mode and self.df.index.str.contains('DBUG').any():
      df = self.remove_debug_links()   # rebuilds self.nodes / self.links too
    else:
      df = self.df

    self.log.info('Creating HiGHS Model (debug=%s)' % debug_mode)
    self.hmodel = HighsNetworkModel(log=self.log)
    self.hmodel.build(df, self.nodes, debug_mode=debug_mode,
                      cosvf_mode=cosvf_mode, debug_cost=debug_cost)
    self._backend = 'highs'
    return self.hmodel

  def solve_highs_model(self, solver='highs', nproc=1, debug_mode=False,
                        maxiter=10, solver_options=None):
    """Solve the HiGHS model (must follow create_highs_model).

    Mirrors solve_pyomo_model: in debug mode it runs the iterative
    bound-relaxation loop (fix_debug_flows_highs) with the same stall/maxiter
    logic; otherwise a single solve that raises RuntimeError on infeasibility.
    """
    opts = {}
    if nproc > 1:
      opts['threads'] = nproc
    if solver_options:
      opts.update(solver_options)
    m = self.hmodel

    if debug_mode:
      run_again = True
      i = 0
      vol_total = 0
      prev_debug = float('inf')
      stalled = False
      while run_again and i < maxiter:
        self.log.info('-----Solving HiGHS Model (debug=%s)' % debug_mode)
        m.solve(need_duals=True, options=opts)
        self.log.info('Finished. Fixing debug flows...')
        run_again, vol, total_debug = self.fix_debug_flows_highs()
        i += 1
        vol_total += vol
        if run_again and total_debug >= prev_debug * 0.99:
          stalled = True
          break
        prev_debug = total_debug

      if not run_again:
        self.log.info('All debug flows eliminated (iter=%d, vol=%0.2f)'
                      % (i, vol_total))
        return True
      elif stalled:
        self.log.warning('Debug stalled at %.2e TAF after %d iterations; '
                         'no further improvement possible.' % (total_debug, i))
        return False
      else:
        self.log.warning('Debug mode maximum iterations reached (%d); '
                         '%.2e TAF of debug flows remain.' % (maxiter, total_debug))
        return False

    self.log.info('-----Solving HiGHS Model (debug=%s)' % debug_mode)
    m.solve(need_duals=True, options=opts)  # raises RuntimeError if infeasible
    self.log.info('Optimal Solution Found (debug=%s).' % debug_mode)
    return True

  def solve_highs_relaxed(self, floors=None, weights=None, solver='highs',
                          nproc=1, solver_options=None):
    """Solve the HiGHS model with targeted feasibility relaxation instead of the
    debug-flow loop (must follow ``create_highs_model(debug_mode=False)``).

    Tries a plain solve first; if the future is infeasible, relaxes only the
    catalogued hard floors (required outflows, instream flows, dead-pool /
    ending storage) by the minimum needed and re-optimizes the economic
    operation on the relaxed network. See :mod:`calvin.relax`.

    :param floors: precomputed catalog (``relaxable_floors(self.df)`` by
      default — built on the current, post-scenario df).
    :param weights: per-category phase-1 penalties (``CATEGORY_WEIGHTS`` default).
    :returns: a :class:`calvin.relax.RelaxSolution` (also stored as
      ``self.relaxation``); the solved flows are read the usual way via
      ``model_to_dataframe`` / ``postprocess``.
    """
    from calvin.relax import relaxable_floors, solve_two_phase, CATEGORY_WEIGHTS

    opts = {}
    if nproc > 1:
      opts['threads'] = nproc
    if solver_options:
      opts.update(solver_options)
    if floors is None:
      floors = relaxable_floors(self.df)
    # neutralize the soft policy penalties (trade overflow, env-flow slack) during
    # phase 1 so the minimal hard-floor relaxation is measured cleanly under dry
    # futures (see solve_two_phase's extra_zero_keys).
    extra_zero_keys = [k for k in self.hmodel.extra_cols
                       if isinstance(k, tuple) and k
                       and k[0] in ('trade_overflow', 'env_slack')]
    sol = solve_two_phase(self.hmodel, floors, weights=weights or CATEGORY_WEIGHTS,
                          options=opts, log=self.log,
                          extra_zero_keys=extra_zero_keys)
    self.relaxation = sol
    if sol.relaxed:
      self.log.info('Relaxed %.2f TAF of hard floors across %d node-months to '
                    'reach feasibility.' % (sol.total_taf, len(sol.report)))
    return sol

  def fix_debug_flows_highs(self, tol=1e-7):
    """HiGHS port of fix_debug_flows: relieve residual DBUG valve flow by raising
    UB on DBUGSNK sources (surplus) or lowering LB on DBUGSRC descendants
    (deficit).  Mutates both the HiGHS column bounds and ``self.df`` (so
    get_bound_adjustments still audits the relaxation).

    :returns: (run_again, vol_total, total_debug) — same contract as
      fix_debug_flows.
    """
    df, m = self.df, self.hmodel
    flows = m.flows()
    bd = m.bound_duals()

    dbix = (df.i.str.contains('DBUGSRC') | df.j.str.contains('DBUGSNK'))
    debuglinks = df[dbix].values

    total_debug = sum(max(flows.get(tuple(dbl[0:3]), 0.0) or 0, 0)
                      for dbl in debuglinks)
    run_again = False
    vol_total = 0

    for dbl in debuglinks:
      s = tuple(dbl[0:3])
      flow_val = flows.get(s, 0.0)
      if flow_val <= tol:
        continue
      run_again = True
      if 'DBUGSNK' in dbl[1]:
        vol_total += self._raise_upper_bounds_highs(dbl, df, m, flow_val)
      elif 'DBUGSRC' in dbl[0]:
        vol_total += self._lower_downstream_bounds_highs(dbl, df, m, flow_val, bd)

    self.df = df
    return run_again, vol_total, total_debug

  def _raise_upper_bounds_highs(self, dbl, df, m, flow_val):
    """Raise UB on outgoing links of a DBUGSNK source node (surplus water)."""
    vol = 0
    raiselinks = df[(df.i == dbl[0]) & ~df.j.str.contains('DBUGSNK')].values
    for l in raiselinks:
      s2 = tuple(l[0:3])
      c = m.arc_index[s2]
      iv = m.col_upper[c]
      v = flow_val * 1.2
      new_ub = iv + v
      m.set_bound(s2, m.col_lower[c], new_ub)
      vol += v
      self.log.info('%s UB raised by %0.2f (%0.2f%%)'
                    % (l[0] + '_' + l[1], v, v * 100 / iv if iv else 0.0))
      df.loc['_'.join(str(x) for x in l[0:3]), 'upper_bound'] = new_ub
    return vol

  def _lower_downstream_bounds_highs(self, dbl, df, m, flow_val, bd, max_depth=10):
    """Reduce LB on downstream links of a DBUGSRC node (deficit water)."""
    vol_to_reduce = max(flow_val * 1.2, 0.5)
    self.log.info('Volume to reduce: %.2e' % vol_to_reduce)

    children = [dbl[1]]
    for _ in range(max_depth):
      children += df[df.i.isin(children) & ~df.j.str.contains('DBUGSNK')].j.tolist()
    children = set(children)

    reducelinks = (df[df.i.isin(children) & (df.lower_bound > 0)]
                   .sort_values(by='lower_bound', ascending=False).values)
    if reducelinks.size == 0:
      raise RuntimeError('Not possible to reduce LB on links with origin %s by '
                         'volume %0.2f' % (dbl[1], vol_to_reduce))

    carryover = ['SR_', 'INITIAL', 'FINAL', 'GW_']
    vol = 0
    for l in reducelinks:
      if vol_to_reduce == 0:
        break
      s2 = tuple(l[0:3])
      c = m.arc_index[s2]
      iv = m.col_lower[c]
      dl = bd.get(s2, (0.0, 0.0))[0]
      if iv > 0 and dl > 1e6:
        v = min(vol_to_reduce, iv)
        if any(x in l[0] for x in carryover) and any(x in l[1] for x in carryover):
          v = min(v, max(25.0, 0.1 * iv))
        new_lb = iv - v
        m.set_bound(s2, new_lb, m.col_upper[c])
        vol_to_reduce -= v
        vol += v
        self.log.info('%s LB reduced by %.2e (%0.2f%%). Dual=%.2e'
                      % (l[0] + '_' + l[1], v, v * 100 / iv, dl))
        df.loc['_'.join(str(x) for x in l[0:3]), 'lower_bound'] = new_lb

    if vol_to_reduce > 0:
      self.log.info('Debug -> %s: could not reduce full amount (%.2e left)'
                    % (dbl[1], vol_to_reduce))
    return vol
