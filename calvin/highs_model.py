"""Direct-to-HiGHS network LP backend for CALVIN (alongside the Pyomo path).

Builds the CALVIN network-flow LP straight into a persistent ``highspy.Highs()``
model, skipping the Pyomo object graph and the LP-file handoff. The mapping:

- one column per arc ``(i, j, k)``; the arc's ``lower_bound``/``upper_bound`` are
  **column bounds** (not constraint rows, the way Pyomo writes them);
- one equality row per node (except SOURCE/SINK) for mass balance, with two
  nonzeros per arc: ``+1`` in the destination row ``j`` (inflow) and ``-1/amplitude``
  in the origin row ``i`` (outflow);
- side constraints (env-flow floors, capacity coupling) are appended as rows and
  capacity variables as columns.

The persistent ``Highs()`` object is kept for warm re-solves with in-place
bound/cost edits, which is what the multi-scenario Benders subproblems and the
fast harness need.

Because the arc bounds are column bounds, the numbers Pyomo reads as
``dual[limit_lower[s]]`` / ``dual[limit_upper[s]]`` come back here as the
variable's **reduced cost** at its lower/upper bound (see :meth:`bound_duals`).
Results are exposed as plain dicts keyed by arc ``(i, j, k)`` / node / column key
so the shared consumers (``postprocess``, ``model_to_dataframe``,
``fix_debug_flows``) can read them without a Pyomo model.
"""

import numpy as np
import pandas as pd

# Nodes that carry no mass-balance row (boundary source/sink of the whole net).
NO_BALANCE = ('SOURCE', 'SINK')


class HighsNetworkModel:
  """Persistent, warm-startable HiGHS build of the CALVIN network LP.

  Typical use::

      m = HighsNetworkModel(log=cal.log)
      m.build(cal.df, cal.nodes, debug_mode=False)
      m.add_rows(specs)              # optional side constraints
      m.solve(need_duals=True)
      flows = m.flows()              # {(i, j, k): flow}
      node_d = m.node_duals()        # {node: marginal water value}
      lo, hi = m.bound_duals()[arc]  # (dual[limit_lower], dual[limit_upper])
  """

  def __init__(self, log=None):
    self.log = log
    self.h = None                 # highspy.Highs() (persistent)
    self.arcs = []                # list of (i, j, k) in column order
    self.arc_index = {}           # (i, j, k) -> base column index
    self.node_row = {}            # node -> mass-balance row index
    self.amp = None               # amplitude per base arc column (np array)
    self.col_lower = None         # mirror of column bounds (np, base arcs)
    self.col_upper = None
    self.col_cost = None          # mirror of objective coeffs (np, base arcs)
    self.n_arc_cols = 0
    self.n_rows = 0
    self.extra_cols = {}          # key -> column index (X_cap, X_exp, ...)
    self.row_labels = {}          # side/coupling row index -> label
    self._debug_cols = None       # column indices of DBUG valve arcs
    self._sol = None              # cached HighsSolution
    self._col_value = None        # np arrays cached at solve time
    self._col_dual = None
    self._row_dual = None
    self._obj = None              # cached objective value
    self._built = False

  # -- logging helper -------------------------------------------------------
  def _log(self, level, msg, *args):
    if self.log is not None:
      getattr(self.log, level)(msg, *args)

  # -- build ----------------------------------------------------------------
  def build(self, df, nodes, debug_mode=False, cosvf_mode=False,
            debug_cost=2e10, with_names=False):
    """Assemble the base network LP from the links DataFrame.

    :param df: links DataFrame (columns i, j, k, cost, amplitude,
      lower_bound, upper_bound). Column order defines arc/column order.
    :param nodes: iterable of node names; row order follows this, minus
      SOURCE/SINK.
    :param debug_mode/cosvf_mode/debug_cost: reproduce the three cost regimes of
      ``CALVIN.create_pyomo_model`` (calvin.py:285-290): debug links get
      ``debug_cost``; plain debug mode flattens real costs to 1.0; cosvf/real
      keep true costs.
    :param with_names: attach column/row names (helpful for IIS on small nets;
      skip on the full model to save memory).
    """
    import highspy

    i_arr = df.i.to_numpy()
    j_arr = df.j.to_numpy()
    k_arr = df.k.to_numpy()
    cost_arr = df.cost.to_numpy(dtype=float)
    amp_arr = df.amplitude.to_numpy(dtype=float)
    lb_arr = df.lower_bound.to_numpy(dtype=float)
    ub_arr = df.upper_bound.to_numpy(dtype=float)
    n = len(df)

    if np.any(amp_arr == 0):
      raise ValueError('zero amplitude on %d arc(s); would divide by zero in '
                       'mass balance' % int(np.sum(amp_arr == 0)))

    self.arcs = list(zip(i_arr, j_arr, k_arr))
    self.arc_index = {a: c for c, a in enumerate(self.arcs)}
    self.amp = amp_arr
    self.n_arc_cols = n

    # -- objective: three cost regimes (mirror calvin.py:285-290) -----------
    is_dbug = np.fromiter(
        ('DBUG' in (str(a) + '_' + str(b)) for a, b in zip(i_arr, j_arr)),
        dtype=bool, count=n)
    if debug_mode:
      base = cost_arr if cosvf_mode else np.ones(n)
      col_cost = np.where(is_dbug, debug_cost, base)
    else:
      col_cost = cost_arr.copy()
    self.col_cost = col_cost
    self.col_lower = lb_arr.copy()
    self.col_upper = ub_arr.copy()
    self._debug_cols = np.nonzero(is_dbug)[0]

    # -- mass-balance rows: one per node except SOURCE/SINK -----------------
    self.node_row = {}
    r = 0
    for node in nodes:
      if node in NO_BALANCE:
        continue
      self.node_row[node] = r
      r += 1
    n_rows = r
    self.n_rows = n_rows

    i_row = np.fromiter((self.node_row.get(x, -1) for x in i_arr),
                        dtype=np.int64, count=n)
    j_row = np.fromiter((self.node_row.get(x, -1) for x in j_arr),
                        dtype=np.int64, count=n)
    has_i = i_row >= 0
    has_j = j_row >= 0

    # Column-major (CSC): per column, the +1 inflow entry (row j) then the
    # -1/amp outflow entry (row i). Interleaving slots 2c, 2c+1 keeps entries
    # grouped in column order, so filtering the invalid (SOURCE/SINK) slots
    # preserves the CSC ordering and start_ follows from nnz-per-column.
    rows2 = np.empty(2 * n, dtype=np.int64)
    vals2 = np.empty(2 * n, dtype=float)
    valid2 = np.empty(2 * n, dtype=bool)
    rows2[0::2] = j_row
    vals2[0::2] = 1.0
    valid2[0::2] = has_j
    rows2[1::2] = i_row
    vals2[1::2] = -1.0 / amp_arr
    valid2[1::2] = has_i

    index = rows2[valid2].astype(np.int32)
    value = vals2[valid2]
    nnz_per_col = has_j.astype(np.int64) + has_i.astype(np.int64)
    starts = np.zeros(n + 1, dtype=np.int32)
    np.cumsum(nnz_per_col, out=starts[1:])

    lp = highspy.HighsLp()
    lp.num_col_ = n
    lp.num_row_ = n_rows
    lp.col_cost_ = col_cost
    lp.col_lower_ = lb_arr.copy()
    lp.col_upper_ = ub_arr.copy()
    lp.row_lower_ = np.zeros(n_rows)
    lp.row_upper_ = np.zeros(n_rows)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = starts
    lp.a_matrix_.index_ = index
    lp.a_matrix_.value_ = value
    lp.sense_ = highspy.ObjSense.kMinimize
    lp.offset_ = 0.0
    if with_names:
      lp.col_names_ = ['%s_%s_%s' % a for a in self.arcs]
      lp.row_names_ = [n for n in nodes if n not in NO_BALANCE]

    h = highspy.Highs()
    h.setOptionValue('output_flag', False)
    h.passModel(lp)
    self.h = h
    self._built = True
    self._log('info', 'HiGHS model built: %d cols, %d rows, %d nonzeros'
              % (n, n_rows, len(value)))
    return self

  # -- side constraints and extra columns -----------------------------------
  def _col(self, key):
    """Resolve an arc tuple or a registered column key to a column index."""
    c = self.arc_index.get(key)
    if c is None:
      c = self.extra_cols.get(key)
    if c is None:
      raise KeyError('unknown column key %r' % (key,))
    return c

  def add_columns(self, specs):
    """Append non-negative columns (capacity vars). Call before the coupling
    rows that reference them.

    :param specs: list of ``(key, lower, upper, cost)``.
    :returns: dict ``{key: column index}`` for the newly added columns.
    """
    import highspy
    if not specs:
      return {}
    ncol = self.h.getNumCol()
    lower = np.array([s[1] for s in specs], dtype=float)
    upper = np.array([s[2] for s in specs], dtype=float)
    cost = np.array([s[3] for s in specs], dtype=float)
    # no matrix entries yet; coupling rows add the coefficients
    starts = np.zeros(len(specs), dtype=np.int32)
    empty_i = np.empty(0, dtype=np.int32)
    empty_v = np.empty(0, dtype=float)
    self.h.addCols(len(specs), cost, lower, upper, 0, starts, empty_i, empty_v)
    added = {}
    for off, s in enumerate(specs):
      self.extra_cols[s[0]] = ncol + off
      added[s[0]] = ncol + off
    return added

  def add_rows(self, specs):
    """Append linear side constraints.

    :param specs: list of ``(coeffs, sense, rhs, label)`` where ``coeffs`` maps a
      column key (arc tuple or registered key) to its coefficient, ``sense`` is
      ``'>='``, ``'<='`` or ``'=='``, ``rhs`` is a float, and ``label`` is any
      hashable tag recorded for dual reporting.
    :returns: list of the new row indices (aligned with ``specs``).
    """
    import highspy
    if not specs:
      return []
    inf = highspy.kHighsInf
    row0 = self.h.getNumRow()
    lowers, uppers, starts, idx, val = [], [], [], [], []
    nz = 0
    for (coeffs, sense, rhs, label) in specs:
      starts.append(nz)
      for key, coef in coeffs.items():
        idx.append(self._col(key))
        val.append(float(coef))
        nz += 1
      if sense == '>=':
        lowers.append(rhs); uppers.append(inf)
      elif sense == '<=':
        lowers.append(-inf); uppers.append(rhs)
      elif sense == '==':
        lowers.append(rhs); uppers.append(rhs)
      else:
        raise ValueError('bad sense %r' % (sense,))
    self.h.addRows(len(specs),
                   np.array(lowers, dtype=float), np.array(uppers, dtype=float),
                   nz, np.array(starts, dtype=np.int32),
                   np.array(idx, dtype=np.int32), np.array(val, dtype=float))
    new_rows = []
    for off, (_, _, _, label) in enumerate(specs):
      ri = row0 + off
      self.row_labels[ri] = label
      new_rows.append(ri)
    return new_rows

  def set_objective_coeff(self, key, cost):
    """Set an objective coefficient on an (already added) column."""
    self.h.changeColCost(self._col(key), float(cost))

  # -- warm edits -----------------------------------------------------------
  def set_bound(self, arc, lower, upper):
    """Change one arc's column bounds (deferred to the next solve; basis kept)."""
    c = self.arc_index[arc]
    self.h.changeColBounds(c, float(lower), float(upper))
    self.col_lower[c] = lower
    self.col_upper[c] = upper

  def set_cost(self, arc, cost):
    """Change one arc's objective coefficient."""
    c = self.arc_index[arc]
    self.h.changeColCost(c, float(cost))
    self.col_cost[c] = cost

  def set_bounds(self, mapping):
    """Batch-change column bounds. ``mapping``: {arc: (lower, upper)}."""
    if not mapping:
      return
    cols = np.array([self.arc_index[a] for a in mapping], dtype=np.int32)
    lo = np.array([v[0] for v in mapping.values()], dtype=float)
    hi = np.array([v[1] for v in mapping.values()], dtype=float)
    self.h.changeColsBounds(len(cols), cols, lo, hi)
    self.col_lower[cols] = lo
    self.col_upper[cols] = hi

  # -- solve ----------------------------------------------------------------
  def solve(self, need_duals=True, options=None, raise_on_infeasible=True):
    """Run HiGHS on the current model (warm from the persistent basis).

    :param need_duals: keep crossover on so reduced costs / row duals are clean.
    :param options: extra ``setOptionValue`` pairs (dict), e.g.
      ``{'solver': 'simplex', 'threads': 4}``.
    :returns: True if optimal. Raises RuntimeError on infeasible unless
      ``raise_on_infeasible`` is False.
    """
    import highspy
    h = self.h
    h.setOptionValue('output_flag', False)
    if need_duals:
      # dual simplex returns a basic solution with clean duals; if a barrier is
      # ever selected via options, crossover must stay on for reliable duals.
      h.setOptionValue('run_crossover', 'on')
    if options:
      for key, v in options.items():
        h.setOptionValue(key, v)

    h.run()
    status = h.getModelStatus()
    if status == highspy.HighsModelStatus.kOptimal:
      sol = h.getSolution()
      self._sol = sol
      # cache as numpy so accessors can fancy-index
      self._col_value = np.asarray(sol.col_value, dtype=float)
      self._col_dual = np.asarray(sol.col_dual, dtype=float)
      self._row_dual = np.asarray(sol.row_dual, dtype=float)
      self._obj = h.getInfo().objective_function_value
      return True

    self._sol = None
    self._obj = None
    self._log('error', 'HiGHS status: %s' % h.modelStatusToString(status))
    if status == highspy.HighsModelStatus.kInfeasible:
      self._log_iis()
    if raise_on_infeasible:
      raise RuntimeError('Problem Infeasible (HiGHS status %s).'
                         % h.modelStatusToString(status))
    return False

  def _log_iis(self):
    """Log the Irreducible Infeasible Subsystem via the resident model."""
    import highspy
    h = self.h
    h.setOptionValue('iis_strategy', int(highspy._core.kIisStrategyIrreducible))
    iis_status, iis = h.getIis()
    if iis_status != highspy.HighsStatus.kOk or not iis.valid_:
      self._log('warning', 'IIS computation failed or returned invalid result.')
      return
    lp = h.getLp()
    col_names, row_names = lp.col_names_, lp.row_names_
    UPPER = int(highspy._core.kIisBoundStatusUpper)
    self._log('error', '--- IIS: %d conflicting bounds, %d conflicting rows ---'
              % (len(iis.col_index_), len(iis.row_index_)))
    for idx, bs in zip(iis.col_index_, iis.col_bound_):
      nm = col_names[idx] if idx < len(col_names) else str(idx)
      self._log('error', '  Variable %s (%s)' % (nm, 'UB' if bs == UPPER else 'LB'))
    for idx, bs in zip(iis.row_index_, iis.row_bound_):
      nm = row_names[idx] if idx < len(row_names) else str(idx)
      self._log('error', '  Constraint %s (%s)' % (nm, 'UB' if bs == UPPER else 'LB'))

  # -- result accessors (plain dicts) ---------------------------------------
  def _require_sol(self):
    if self._col_value is None:
      raise RuntimeError('no solution; call solve() first (and confirm optimal)')

  def objective(self):
    return self._obj

  def flows(self):
    """{(i, j, k): flow} over the base arcs."""
    self._require_sol()
    cv = self._col_value
    return {a: cv[c] for a, c in self.arc_index.items()}

  def cap_values(self):
    """{key: value} over the extra (capacity) columns."""
    self._require_sol()
    cv = self._col_value
    return {key: cv[c] for key, c in self.extra_cols.items()}

  def node_duals(self):
    """{node: dual of its mass-balance row} — the marginal water value."""
    self._require_sol()
    rd = self._row_dual
    return {node: rd[r] for node, r in self.node_row.items()}

  def bound_duals(self):
    """{(i, j, k): (dual[limit_lower], dual[limit_upper])} from reduced costs.

    With column bounds, HiGHS' reduced cost is the shadow price of whichever
    bound is active. For minimization it is >= 0 at the lower bound and <= 0 at
    the upper bound. Pyomo's ``dual[limit_lower]`` (an ``X >= l`` row) is the
    positive part; ``dual[limit_upper]`` (an ``X <= u`` row) is the negative
    part. (Signs verified against the Pyomo path in test_highs_parity.py.)
    """
    self._require_sol()
    cd = self._col_dual
    out = {}
    for a, c in self.arc_index.items():
      rc = cd[c]
      out[a] = (rc if rc > 0 else 0.0, rc if rc < 0 else 0.0)
    return out

  def coupling_duals(self):
    """{label: row dual} over the side/coupling rows added via add_rows."""
    self._require_sol()
    rd = self._row_dual
    return {label: rd[ri] for ri, label in self.row_labels.items()}

  def residual_debug(self):
    """Total flow on DBUG valve arcs (the harness feasibility oracle)."""
    if self._debug_cols is None or len(self._debug_cols) == 0:
      return 0.0
    self._require_sol()
    vals = self._col_value[self._debug_cols]
    return float(np.sum(np.clip(vals, 0.0, None)))

  def to_dataframe(self):
    """Per-link DataFrame (flow, cost, lower_bound, upper_bound), matching
    ``CALVIN.model_to_dataframe``. Bounds/costs come from the current mirrors
    (so they reflect any warm bound relaxation)."""
    self._require_sol()
    cv = self._col_value
    recs = [(a[0], a[1], a[2], cv[c], self.col_cost[c],
             self.col_lower[c], self.col_upper[c])
            for a, c in self.arc_index.items()]
    mdf = pd.DataFrame(recs, columns=['i', 'j', 'k', 'flow', 'cost',
                                      'lower_bound', 'upper_bound'])
    mdf['link'] = mdf.i.map(str) + '-' + mdf.j.map(str) + '-' + mdf.k.map(str)
    mdf.set_index('link', inplace=True)
    return mdf
