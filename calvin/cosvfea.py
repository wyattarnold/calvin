import os, json, pickle
import logging
import tempfile, shutil
import numpy as np
import pandas as pd
import math, copy, random
from sympy import Symbol, factorial, nsolve
from deap import algorithms, creator, tools
from deap.base import Fitness
from deap.base import Toolbox
from .calvin import *
from .postprocessor import *
from .postprocessor import _collect_links

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Solver helpers
# ---------------------------------------------------------------------------

def _init_cosvf_solver(solver, nproc, log):
    """Initialize solver for the COSVF annual sequence.

    For Gurobi, CPLEX, and HiGHS, attempts to use Pyomo's APPSI persistent
    interface.  APPSI keeps the LP resident in the solver's in-memory
    representation between years and only pushes changed bounds/costs as
    deltas, eliminating LP-file I/O and full model re-builds on each year.

    Falls back to a standard ``SolverFactory`` for all other solvers (CBC,
    GLPK) and whenever APPSI is unavailable.

    Returns
    -------
    opt
        Solver object (APPSI or ``SolverFactory`` instance).
    is_appsi : bool
        ``True`` if *opt* is an APPSI persistent solver.
    """
    _appsi_map = {'gurobi': 'Gurobi', 'cplex': 'Cplex', 'highs': 'Highs'}

    if solver in _appsi_map:
        try:
            from pyomo.contrib import appsi
            opt = getattr(appsi.solvers, _appsi_map[solver])()
            if opt.available():
                log.info('COSVF: using APPSI persistent interface for %s', solver)
                # Dual simplex is ideal for warm-started problems where only
                # bounds change between solves (primal stays feasible).
                if solver == 'gurobi':
                    opt.gurobi_options['Method'] = 1     # dual simplex
                    if nproc > 1:
                        opt.gurobi_options['Threads'] = nproc
                elif solver == 'cplex':
                    opt.cplex_options['lpmethod'] = 2    # dual simplex
                    if nproc > 1:
                        opt.cplex_options['threads'] = nproc
                elif solver == 'highs':
                    opt.highs_options['simplex_strategy'] = 3  # dual simplex
                    opt.highs_options['threads'] = nproc       # always pin threads; prevents HiGHS
                                                               # from auto-detecting all CPUs on the
                                                               # machine when nproc=1 in multiworker runs
                return opt, True
        except Exception as exc:
            log.debug('APPSI not available for %s (%s); using SolverFactory', solver, exc)

    # Standard file-based SolverFactory (CBC, GLPK, or fallback)
    from pyomo.opt import SolverFactory
    opt = SolverFactory(solver)
    if nproc > 1 and solver != 'glpk':
        opt.options['threads'] = nproc
    log.info('COSVF: using SolverFactory for %s', solver)
    return opt, False


class _PostprocessCollector:
    """Accumulate COSVF postprocess data across all water years; write CSVs once.

    Replaces 82 × N per-year file-open/close cycles with a single
    ``save_dict_as_csv`` call per output file after the annual
    loop completes.  Lookup CSVs (demand_nodes, pwp_nodes, operation_nodes) are
    loaded once in ``__init__`` rather than once per year.
    """

    def __init__(self):
        self.F, self.S, self.E = {}, {}, {}
        self.SV, self.SC = {}, {}
        self.PV, self.PC, self.OC = {}, {}, {}
        self.D_up, self.D_lo, self.D_node = {}, {}, {}
        self.EOP_storage = {}

        # Load lookup tables once (avoid repeated file reads inside the loop)
        _demand = pd.read_csv(os.path.join(BASE_DIR, 'data', 'demand_nodes.csv'), index_col=0)
        _pwp    = pd.read_csv(os.path.join(BASE_DIR, 'data', 'pwp_nodes.csv'),    index_col=0)
        _op     = pd.read_csv(os.path.join(BASE_DIR, 'data', 'operation_nodes.csv'), index_col=0)
        self._demand_set = set(_demand.index)
        self._pwp_set    = set(_pwp.index)
        self._op_set     = set(_op.index)

    def collect(self, df, model, year):
        """Collect one water year — delegates link accumulation to _collect_links."""
        links = df.values
        nodes = pd.unique(df[['i', 'j']].values.ravel()).tolist()

        _collect_links(
            links, model, year,
            self.F, self.S, self.E, self.SV, self.SC, self.PV, self.PC, self.OC,
            self.D_up, self.D_lo, self.EOP_storage,
            self._demand_set, self._pwp_set, self._op_set,
        )

        for node in nodes:
            if '.' in node:
                n3, t3 = node.split('.')
                d3 = model.dual.get(model.flow[node], 0.0) if node in model.flow else 0.0
                if year is not None and year > 1922:
                    t3 = t3.replace('1922', str(year)).replace('1921', str(year - 1))
                dict_insert(self.D_node, n3, t3, d3)

    def write(self, resultdir):
        """Write all accumulated data to CSV — one file-open per output, mode='w'."""
        os.makedirs(resultdir, exist_ok=True)
        things = [
            (self.F,           'flow'),
            (self.S,           'storage'),
            (self.D_up,        'dual_upper'),
            (self.D_lo,        'dual_lower'),
            (self.D_node,      'dual_node'),
            (self.E,           'evaporation'),
            (self.SV,          'shortage_volume'),
            (self.SC,          'shortage_cost'),
            (self.EOP_storage, 'eop_storage'),
            (self.PV,          'pwp_short_volume'),
            (self.PC,          'pwp_short_cost'),
            (self.OC,          'operation_costs'),
        ]
        for data, name in things:
            if data:
                save_dict_as_csv(data, os.path.join(resultdir, name + '.csv'))


###############################################################################
### Limited foresight Carryover storage value function (COSVF) CALVIN model ###
###############################################################################
class COSVF(CALVIN):
  
  def __init__(self, pwd, log_name="calvin-cosvf", console_level=logging.INFO):
    """
    Instantiate COSVF model as a child class of ``calvin.CALVIN`` for annual COSVF optimization.

    :param pwd: (string) path to directory containing COSVF input files.  Generate these
      files with :func:`calvin.network.prepare.prepare_cosvf` or the CLI equivalent::

        python -m calvin.network.cli prepare-cosvf \\
            --data /path/to/calvin-network-data/data \\
            --output ./my-models/calvin-cosvf

      Required files in the directory:

      1. ``links.csv`` — single water-year network matrix
         (``i,j,k,cost,amplitude,lower_bound,upper_bound``)
      2. ``cosvf-params.csv`` — penalty parameters (``r,param,value``)
      3. ``r-dict.json`` — reservoir dictionary with penalty properties
      4. ``inflows.csv`` — external inflows for the full period (``date,j,flow_taf``)
      5. ``variable-constraints.csv`` — time-varying link bounds
         (``date,i,j,k,lower_bound,upper_bound``)

      See :func:`calvin.network.prepare.prepare_cosvf` for details on each file.

    :param log_name: (string) name for the global logger. Log file is written to the specified ``pwd`` path.
    :param console_level: (int) logging level for the console handler (default ``logging.INFO``).
      Pass ``logging.WARNING`` in EA worker processes to suppress per-solve console output
      while still writing full DEBUG detail to the log file.
    :returns: COSVF CALVIN model object
    """
    # set working directory
    self.pwd = pwd

    # check that required input files exist
    linksfile = os.path.join(self.pwd, 'links.csv')
    if not os.path.isfile(linksfile):
      raise FileNotFoundError(
        f"COSVF input files not found in {self.pwd}. "
        "Generate them first with: python -m calvin.network.cli prepare-cosvf "
        "--data /path/to/calvin-network-data/data --output " + self.pwd
      )

    # set up logging code
    self.log = setup_logger(log_name, savedir=pwd, console_level=console_level)

    self._load_input_files()

    # a few network fixes to make things work
    super().add_ag_region_sinks()

    # lists for unique nodes and links
    self.nodes = pd.unique(self.df[['i','j']].values.ravel()).tolist()
    self.links = list(zip(self.df.i,self.df.j,self.df.k))

    # make sure things aren't broken
    super().networkcheck()

    # precompute static link sets for fast per-year fitness evaluation
    self._precompute_fitness_links()

    # precompute model-update data (inflow keys/matrix, VC entries, EOP keys)
    self._precompute_update_data()


  def _load_input_files(self):
    """Load all COSVF input files from self.pwd and build the link dataframe."""
    self.df = pd.read_csv(os.path.join(self.pwd, 'links.csv'))

    with open(os.path.join(self.pwd, 'r-dict.json')) as f:
      self.r_dict = json.load(f)
    self.nrtype1 = sum(1 for v in self.r_dict.values() if v.get('type') == 1)
    self.nrtype2 = sum(1 for v in self.r_dict.values() if v.get('type') == 2)

    self.pcosvf = np.loadtxt(os.path.join(self.pwd, 'cosvf-params.csv'),
                              delimiter=',', skiprows=1, usecols=2).tolist()

    inflows = pd.read_csv(os.path.join(self.pwd, 'inflows.csv'), index_col=0, parse_dates=True)
    self.inflows = inflows.pivot(columns='j', values='flow_taf').rename_axis(None, axis=1)
    self.wy_start = int(min(self.inflows.index.year)) + 1
    self.wy_end = int(max(self.inflows.index.year))
    self.inflow_terminals = inflows.j.unique()

    self.variable_constraints = pd.read_csv(
      os.path.join(self.pwd, 'variable-constraints.csv'), index_col=0, parse_dates=True)

    self.create_cosvf_links()
    self.df['link'] = self.df.i.map(str) + '_' + self.df.j.map(str) + '_' + self.df.k.map(str)
    self.df.set_index('link', inplace=True)


  def create_cosvf_links(self):
    """
    Create k-links for the storage nodes that define the carryover penalties. 

    :returns: nothing, but modifies links dataframe
    """
    df, r_dict = self.df, self.r_dict

    # loop through reservoirs to construct piecewise COSVF placeholders
    for r in r_dict:
      # for reservoir w/ penalties
      if r_dict[r]['type']>=1:

        # edit k, ub, and lb in calvin r final nodes
        l = df[(df.i.str.contains(r)) & (df.j.str.contains('FINAL'))].copy()

        # remove r node so it's not duplicated
        self.df.drop(l.index, inplace=True) 

        # add in the minimum capacity for k=0
        l.lower_bound, l.upper_bound, l.cost = r_dict[r]['lb'], r_dict[r]['ub'], 0.0

        # add k-links
        l = pd.concat([l]*(r_dict[r]['k_count']), ignore_index=True)
        l.loc[:,'k'] = list(range(r_dict[r]['k_count']))

        # provide dummy penalty link costs and breakpoints (will be replaced by solve init)
        l.loc[l.k > 0,'lower_bound'] = 0
        l.loc[l.k > 0,'upper_bound'] = 0
        l.loc[l.k==0,'cost'] = -0.01
      
        self.df = pd.concat([self.df, l], ignore_index=False)


  def _precompute_fitness_links(self):
    """Precompute link sets for direct fitness computation in the annual COSVF loop.

    Called once at the end of ``__init__`` so that the per-year
    :meth:`_compute_fitness_direct` call reads ``model.X`` / ``model.u``
    directly without constructing DataFrames or applying pandas filters.

    Replicates the filter logic in :meth:`compute_network_costs` and
    :meth:`compute_gw_overdraft` against the static ``self.df`` snapshot.
    Variable-constraint updates affect ``model.u`` at solve time; only the
    *membership* of each link in a category is static.

    Populates
    ---------
    _shortage_links : list of (i, j, k, cost)
        Links with cost < 0 and static UB < 1e6 (demand shortage links),
        excluding COSVF penalty links, SR→SR persuasions, and DBUG links.
    _op_links : list of (i, j, k, cost)
        Links with cost > 0 (operational costs), same exclusions.
    _gw_final_links : list of (i, j, k)
        GW reservoir → FINAL links with cost < 0 (overdraft measure).
    """
    df = self.df

    # Exclusion mask — identical to compute_network_costs()
    is_sr_gw_final = (
        (df.i.str.contains('SR') | df.i.str.contains('GW')) &
        df.j.str.contains('FINAL')
    )
    is_sr_sr = df.i.str.contains('SR') & df.j.str.contains('SR')
    is_dbug  = df.index.str.contains('DBUG')
    cost_df  = df[~(is_sr_gw_final | is_sr_sr | is_dbug)]

    # Shortage links (cost < 0, static UB < 1e6 to exclude channel persuasions)
    short_mask = (cost_df.cost < 0) & (cost_df.upper_bound < 1e6)
    self._shortage_links = [
        (row.i, row.j, int(row.k), float(row.cost))
        for _, row in cost_df[short_mask].iterrows()
    ]

    # Operational cost links (cost > 0)
    op_mask = cost_df.cost > 0
    self._op_links = [
        (row.i, row.j, int(row.k), float(row.cost))
        for _, row in cost_df[op_mask].iterrows()
    ]

    # GW final links for overdraft calculation
    gw_mask = df.i.str.contains('GW_') & (df.j == 'FINAL') & (df.cost < 0)
    self._gw_final_links = [
        (row.i, row.j, int(row.k))
        for _, row in df[gw_mask].iterrows()
    ]


  def _precompute_update_data(self):
    """Precompute all data for the fast annual model-update path."""
    self._precompute_inflow_data()
    self._precompute_vc_data()
    self._precompute_storage_keys()

  def _precompute_inflow_data(self):
    """Precompute _inflow_keys and _inflow_matrix for fast per-year inflow updates."""
    n_years  = self.wy_end - self.wy_start + 1
    wy_start = self.wy_start
    terminals  = list(self.inflows.columns)
    tdates     = pd.date_range('{}-10-31'.format(wy_start - 1),
                               '{}-09-30'.format(wy_start), freq='ME')
    tdate_strs = [str(d.date()) for d in tdates]

    # Pyomo key tuples for every (terminal, month) combination
    self._inflow_keys = [
        ('INFLOW.{}'.format(fd), '{}.{}'.format(t, fd))
        for t in terminals
        for fd in tdate_strs
    ]

    # Inflow value matrix: (n_years, n_terminals * 12)
    # Rows of self.inflows are monthly dates, columns are terminals.
    # reshape → (n_years, 12, n_terminals); transpose → (n_years, n_terminals, 12);
    # flatten last two dims → matches key order (terminals-outer, months-inner).
    all_dates = pd.date_range('{}-10-31'.format(wy_start - 1),
                              '{}-09-30'.format(self.wy_end), freq='ME')
    arr = self.inflows.loc[all_dates, terminals].values.reshape(n_years, 12, len(terminals))
    self._inflow_matrix = arr.transpose(0, 2, 1).reshape(n_years, len(terminals) * 12)

  def _precompute_vc_data(self):
    """Precompute _vc_by_year: VC entries grouped by water year with template keys."""
    wy_start = self.wy_start
    vc = self.variable_constraints

    # Vectorised template-key computation on the entire DataFrame at once.
    # Template rule: month >= 10 → use (wy_start − 1), else → use wy_start.

    # i-key: 'NODE.YYYY-MM-DD' → 'NODE.TYYYY-MM-DD'
    # pd.DateOffset clamps Feb 29 → Feb 28 in non-leap template years; replicate
    # that by fixing the MM-DD suffix before string concatenation.
    i_split  = vc['i'].str.split('.', expand=True)
    i_node   = i_split[0]
    i_dpart  = i_split[1]
    i_month  = i_dpart.str[5:7].astype(int)
    i_tyear  = np.where(i_month >= 10, wy_start - 1, wy_start)
    i_mmdd   = i_dpart.str[5:].str.replace('02-29', '02-28', regex=False)
    i_tkey   = (i_node + '.' +
                pd.Series(i_tyear.astype(str), index=i_node.index) + '-' + i_mmdd)

    # j-key: some rows have 'FINAL' (no dot); keep those unchanged
    j_has_dot = vc['j'].str.contains(r'\.', regex=True)
    j_split   = vc['j'].str.split('.', expand=True)
    j_node    = j_split[0]
    j_dpart   = j_split[1].fillna('')
    j_month   = j_dpart.str[5:7].replace('', '0').astype(int)
    j_tyear   = np.where(j_month >= 10, wy_start - 1, wy_start)
    j_mmdd    = j_dpart.str[5:].str.replace('02-29', '02-28', regex=False)
    j_computed = (j_node + '.' +
                  pd.Series(j_tyear.astype(str), index=j_node.index) + '-' + j_mmdd)
    j_tkey    = np.where(j_has_dot, j_computed.values, vc['j'].values)

    idx    = pd.DatetimeIndex(vc.index)
    wy_arr = np.where(idx.month >= 10, idx.year + 1, idx.year)

    vc_ext = pd.DataFrame({
        '_ki': i_tkey.values,
        '_kj': j_tkey,
        '_k':  vc['k'].values.astype(int),
        '_lb': vc['lower_bound'].values.astype(float),
        '_ub': vc['upper_bound'].values.astype(float),
        '_wy': wy_arr,
    })

    # Build a frozenset of valid template link keys so that VC rows whose
    # computed template key does not exist in the model are silently dropped.
    # The common case is cross-year storage links (e.g. SR_BUC.YYYY-09-30 →
    # SR_BUC.YYYY-10-31) that map to a backwards template key because the
    # j-month (10) is assigned to wy_start-1 while i-month (9) is wy_start.
    # In the COSVF template these carry-over links are replaced by EOP
    # (→ FINAL) links and therefore do not exist in self.df.
    _valid_links = frozenset(zip(self.df.i, self.df.j, self.df.k.astype(int)))

    self._vc_by_year = {
        int(wy): [
            (ki, kj, k, lb, ub)
            for ki, kj, k, lb, ub in zip(
                grp['_ki'], grp['_kj'], grp['_k'], grp['_lb'], grp['_ub']
            )
            if (ki, kj, k) in _valid_links
        ]
        for wy, grp in vc_ext.groupby('_wy')
    }

  def _precompute_storage_keys(self):
    """Precompute _initial_storage_keys and _eop_keys for reservoir IC and EOP updates."""
    wy_start = self.wy_start
    self._initial_storage_keys = {
        r: ('INITIAL', '{}.{}-10-31'.format(r, wy_start - 1), 0)
        for r in self.r_dict
    }
    self._eop_keys = {
        r: [
            ('{}.{}-09-30'.format(r, wy_start), 'FINAL', k)
            for k in range(self.r_dict[r]['k_count'])
        ]
        for r in self.r_dict
    }


  def _compute_fitness_direct(self):
    """Compute fitness values directly from the solved Pyomo model.

    Replaces :meth:`model_to_dataframe` + :meth:`compute_network_costs` +
    :meth:`compute_gw_overdraft` with a single pass over the precomputed
    link lists, reading ``model.X[s].value`` (flow) and ``model.u[s].value``
    (upper bound, which reflects any variable-constraint updates for the
    current water year) without constructing DataFrames.

    Returns
    -------
    short_costs : float
        Total shortage penalty cost across all shortage links (\\$/yr).
    op_costs : float
        Total operational cost across all op links (\\$/yr).
    gw_overdraft : float
        Total groundwater overdraft volume (TAF) — sum of (UB − flow) for
        GW→FINAL links where flow < UB.
    """
    model = self.model
    short_costs = 0.0
    op_costs    = 0.0

    for (i, j, k, cost) in self._shortage_links:
      s    = (i, j, k)
      flow = model.X[s].value if s in model.X else 0.0
      ub   = model.u[s].value
      shortage = ub - flow
      if shortage > 0:
        short_costs += -cost * shortage   # cost < 0, so -cost > 0

    for (i, j, k, cost) in self._op_links:
      s    = (i, j, k)
      flow = model.X[s].value if s in model.X else 0.0
      op_costs += cost * flow

    gw_overdraft = 0.0
    for (i, j, k) in self._gw_final_links:
      s      = (i, j, k)
      flow   = model.X[s].value if s in model.X else 0.0
      ub     = model.u[s].value
      change = flow - ub
      if change < 0:
        gw_overdraft += -change

    return short_costs, op_costs, gw_overdraft


  def create_pyomo_model(self, **kwargs):
    """
    Create the pyomo model for COSVF mode.

    The COSVF instance of CALVIN uses CALVIN's ``create_pyomo_model`` but with ``cosvf_mode`` parameter **always** on.
    The only difference is whether debug links will be used or not. When debug_mode is used with COSVF, the
    debug links are assigned the default (or user specified) ```debug_cost`` of 2e7 \\$/af; however, all other cost links
    are left with costs as is. See ``calvin.create_pyomo_model`` ``init_params`` function.

    :returns: nothing
    """
    super().create_pyomo_model(cosvf_mode=True, **kwargs)


  def _capture_initial_storage(self):
    """Snapshot initial storage l/u bounds after create_pyomo_model.

    Call once after :meth:`create_pyomo_model` to enable worker-process reuse
    via :meth:`_reset_initial_storage`.  The snapshot preserves the starting
    storage levels so that :meth:`cosvf_solve_reuse` can restore them before
    each new individual evaluation without rebuilding the model.
    """
    from pyomo.core import value as _pyoval
    self._initial_storage_snapshot = {
        r: (_pyoval(self.model.l[key]), _pyoval(self.model.u[key]))
        for r, key in self._initial_storage_keys.items()
    }


  def _reset_initial_storage(self):
    """Restore initial storage to the values captured by _capture_initial_storage.

    Called at the start of :meth:`cosvf_solve_reuse` to reset model state
    before each evaluation in a worker process that reuses the Pyomo model.
    """
    for r, key in self._initial_storage_keys.items():
      l_val, u_val = self._initial_storage_snapshot[r]
      self.model.l[key] = l_val
      self.model.u[key] = u_val


  def _prepare_cosvf_solve(self, solver, nproc, pcosvf, resultdir):
    """Initialize solver, penalties, and postprocess collector for cosvf_solve.

    Returns (opt, _appsi, _basis_dir, _basis_file, _collector, f3).
    """
    opt, _appsi = _init_cosvf_solver(solver, nproc, self.log)

    # CBC: warm-start basis file + dual simplex + disable presolve after year 1
    _basis_dir, _basis_file = None, None
    if not _appsi and solver == 'cbc':
      opt.options['dualSimplex'] = ''
      _basis_dir = tempfile.mkdtemp(prefix='calvin-cosvf-')
      _basis_file = os.path.join(_basis_dir, 'warm.bas')
      opt.options['basisOut'] = _basis_file
    elif not _appsi and solver == 'cplex':
      opt.options['lpmethod'] = 2
    elif not _appsi and solver == 'gurobi':
      opt.options['Method'] = 1

    if pcosvf is not None:
      self.pcosvf = pcosvf
    self.assign_cosvf_penalties()
    # Only write the reference CSV when postprocessing (resultdir is not None);
    # skip in EA mode to avoid 80 workers racing to write the same file.
    if resultdir is not None:
      super().model_to_dataframe().to_csv(
        os.path.join(self.pwd, 'links-pyomo-model-reference.csv'))

    f3 = np.mean(np.array(self.pcosvf)) * -1
    _collector = _PostprocessCollector() if resultdir is not None else None
    return opt, _appsi, _basis_dir, _basis_file, _collector, f3


  def cosvf_solve_reuse(self, opt, appsi, pcosvf, solver='highs'):
    """Evaluate one EA individual using a pre-initialized solver (worker reuse mode).

    Skips model construction and solver initialisation, which dominate runtime
    when workers rebuild everything per call.  Instead:

    1. Resets initial storage to the snapshot from :meth:`_capture_initial_storage`.
    2. Assigns new COSVF penalty parameters.
    3. Runs the full annual sequence using the cached solver.

    Prerequisites
    -------------
    Call :meth:`create_pyomo_model` then :meth:`_capture_initial_storage` once
    before the first call (done in the worker process initialiser).

    Parameters
    ----------
    opt : APPSI solver or SolverFactory instance
        Pre-initialised solver (returned by :func:`_init_cosvf_solver`).
    appsi : bool
        ``True`` if *opt* is an APPSI persistent solver.
    pcosvf : list
        COSVF penalty parameters for this individual.
    solver : str, optional
        Solver name used for CBC basis-file warm-start logic.  Default
        ``'highs'`` (APPSI mode; no basis file needed).

    Returns
    -------
    tuple of (f1, f2, f3) fitness values
    """
    self._reset_initial_storage()

    if pcosvf is not None:
      self.pcosvf = pcosvf
    self.assign_cosvf_penalties()

    f3 = np.mean(np.array(self.pcosvf)) * -1

    f1, f2, years = 0, 0, 1
    for wy in range(self.wy_start, self.wy_end + 1):
      first_year = (wy == self.wy_start)
      f1_inc, f2, ok = self._solve_one_year(
        wy, first_year, opt, appsi, solver, None, None)
      if not ok:
        self.log.info('Solver issue! Fitness values set to infinite')
        return (np.inf, np.inf, np.inf)
      f1 += f1_inc
      years = wy - self.wy_start + 1

    return f1 / 1e3 / years, f2 / 1e3 / years, f3


  def _solve_one_year(self, wy, first_year, opt, _appsi, solver, _basis_file, _collector):
    """Update model for one water year, solve, and return (f1_increment, f2, is_optimal).

    f2 (GW overdraft) is not accumulated across years — the last year's value
    reflects cumulative overdraft from initial conditions.
    """
    from pyomo.opt import TerminationCondition

    if not first_year:
      eop = {r: sum(self.model.X[s].value for s in keys)
             for r, keys in self._eop_keys.items()}
      self.cosvf_update_initial_storage(eop=eop)

    self.cosvf_update_inflows(wy=wy)
    if self.variable_constraints is not None:
      self.cosvf_update_variable_bounds(wy=wy)

    # CBC: from year 2 onward, warm-start from the previous basis and skip presolve
    if solver == 'cbc' and not first_year and _basis_file and os.path.exists(_basis_file):
      opt.options['basisIn'] = _basis_file
      opt.options['presolve'] = 'off'

    self.log.debug('-----Solving Pyomo Model (wy=%d)' % wy)
    if _appsi:
      opt.config.load_solution = False
      self.results = opt.solve(self.model)
      tc_str = str(self.results.termination_condition).lower()
      is_optimal = 'optimal' in tc_str and 'infeasible' not in tc_str
      if is_optimal:
        self.results.solution_loader.load_vars()
    else:
      self.results = opt.solve(self.model, keepfiles=False)
      is_optimal = (
        self.results.solver.termination_condition == TerminationCondition.optimal
      )

    if not is_optimal:
      return 0.0, 0.0, False

    if not _appsi:
      self.model.solutions.load_from(self.results)
    elif _collector is not None:
      # APPSI does not auto-populate model.dual; load explicitly when postprocessing.
      for con, val in self.results.solution_loader.get_duals().items():
        self.model.dual[con] = val

    short_costs, op_costs, f2 = self._compute_fitness_direct()
    if _collector is not None:
      _collector.collect(self.df, self.model, wy)
    return short_costs + op_costs, f2, True

  def cosvf_solve(self, solver='highs', nproc=1, resultdir=None, pcosvf=None, show_progress=False):
    """
    Solve COSVF CALVIN model for full period of analysis

    :param solver: (string) solver name. glpk, cplex, cbc, gurobi, highs.
    :param nproc: (int) number of processors assigned to model solver instance
    :param resultdir: (path) directory to write out results. If ``None`` (default), the assumption
      is that the user is running in evolutionary mode
    :param pcosvf: (list) If ``None`` (default) the COSVF parameters loaded when constructing
      the COSVF CALVIN instance (``cosvf-params.csv``) will be used. Otherwise,
      and specifically for evolutionary mode, the argument is the list of :math:`P_{min}`
      and :math:`P_{max}` for quadratic carryover penalty curves on surface water
      reservoirs and :math:`P_{GW}` for linear penalty on groundwater reservoirs, where
      the order of the penalty parameters for each reservoir must match the
      order of reservoirs in the ``r_dict.json``.
    :param show_progress: (bool) display a tqdm progress bar in the console (default False).
      Requires ``tqdm`` to be installed; silently disabled if it is not.
    :returns: tuple of (f1, f2, f3) fitness values

    **Performance notes**

    - *Gurobi / CPLEX / HiGHS*: automatically uses Pyomo's APPSI persistent
      interface, which keeps the LP in the solver's memory between years and
      pushes only changed bounds as deltas.  Dual simplex is selected so that
      the previously feasible basis is re-optimised rather than restarted from
      scratch.
    - *CBC*: uses the file-based interface with warm-start basis handoff.
      After the first year the previous optimal basis is fed back (``basisIn``),
      LP presolve is disabled (``presolve off``), and dual simplex is selected
      (``dualSimplex``).  Together these typically halve solve time on years 2–82
      versus cold-starting with primal simplex.
    - *GLPK*: unchanged (GLPK does not support basis warm-start via Pyomo).
    """
    opt, _appsi, _basis_dir, _basis_file, _collector, f3 = \
      self._prepare_cosvf_solve(solver, nproc, pcosvf, resultdir)

    _n_years = self.wy_end - self.wy_start + 1
    if show_progress:
      try:
        from tqdm import tqdm
        _year_iter = tqdm(range(self.wy_start, self.wy_end + 1),
                         total=_n_years, unit='yr',
                         bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} yr '
                                    '[{elapsed}<{remaining}, {rate_fmt}]{postfix}')
      except ImportError:
        self.log.warning('tqdm not installed; progress bar disabled')
        _year_iter = range(self.wy_start, self.wy_end + 1)
        show_progress = False
    else:
      _year_iter = range(self.wy_start, self.wy_end + 1)

    f1, f2, years = 0, 0, 1
    _completed = False
    try:
      for wy in _year_iter:
        years = wy - self.wy_start + 1
        first_year = (wy == self.wy_start)
        f1_inc, f2, ok = self._solve_one_year(
          wy, first_year, opt, _appsi, solver, _basis_file, _collector)
        if not ok:
          self.log.info('Solver issue! Fitness values set to infinite')
          return (np.inf, np.inf, np.inf)
        f1 += f1_inc
        self.log.debug('Costs \\$M/yr=%.1f; GW O.D. MAF/yr=%.1f' % (f1/1e3/years, f2/1e3/years))
        if show_progress:
          _year_iter.set_postfix(
            wy=wy,
            cost='${:.0f}M'.format(f1 / 1e3 / years),
            gw_od='{:.2f}MAF'.format(f2 / 1e3 / years),
          )
      _completed = True
    finally:
      if _basis_dir:
        shutil.rmtree(_basis_dir, ignore_errors=True)
      if _collector is not None and _completed:
        _collector.write(resultdir)

    return f1 / 1e3 / years, f2 / 1e3 / years, f3


  def compute_network_costs(self, model_df):
    """
    Calculate costs of LF model run for evolutionary alogrithm.

    :param model_df: (Pandas dataframe) dataframe of cost, upper bound, and flows
      from the solved CALVIN instance
    :returns short_costs: (float) total costs for shorted links
    :returns op_costs: (float) total costs over operational links
    """
    # drop COSVF storage links since not included in total cost fitness
    cost_links = model_df.drop(model_df[((model_df['i'].str.contains('SR')) |
                                        (model_df['i'].str.contains('GW'))) &
                                        (model_df['j'].str.contains('FINAL'))].index)
    # drop storage persuasion penalties (SR→SR self-links)
    cost_links = cost_links.drop(cost_links[(cost_links['i'].str.contains('SR')) &
                                        (cost_links['j'].str.contains('SR'))].index)                                   
    cost_links = cost_links.loc[~cost_links.index.str.contains('DBUG')]

    # all shortage cost links
    short_links = cost_links.loc[(cost_links['cost']<0)]
    short_links = short_links.loc[short_links.upper_bound < 1e6] # drop channel persuasions
    short_costs = -1 * ((short_links.upper_bound - short_links.flow) * short_links.cost).sum()

    # all op cost links
    op_links = cost_links.loc[(cost_links['cost']>0)]
    op_costs = (op_links.flow * op_links.cost).sum()

    return short_costs, op_costs


  def compute_gw_overdraft(self, model_df):
    """
    Calculate overdraft of all groundwater reservoirs that have costs
    
    :param model_df: (Pandas dataframe) dataframe of cost, upper bound, and flows
      from the solved annual Pyomo CALVIN instance
    :returns: (float) total groundwater overdraft of all groundwater reservoirs
    """
    # get groundwater reservoir final links from model
    gw = model_df.loc[(model_df.index.str.contains('GW_')) & 
                            (model_df.index.str.contains('FINAL') &
                            (model_df.cost<0))]

    # calculate groundwater volume change
    gw_change = gw.flow-gw.upper_bound

    # query out overdrafted gw reservoirs and calculate total overdraft
    gw_od = gw_change.iloc[np.where(gw_change<0)]
    gw_total_od = (-1*gw_od).sum()
    
    return gw_total_od


  def cosvf_fit_from_params(self, pmin, pmax, eop_min, eop_max, k_count):
    """
    Determine piecewise costs for COSVF

    :param pmin: (float) penalty representing willingness to pay 
                  for an additional unit of storage that would encroach the
                  rain-flood conservation pool
    :param pmax: (float) penalty representing willingness to pay 
                  for an additional unit of storage below the minimum operating bound
    :param eop_min: (float) end-of-year storage minimum bound
    :param eop_max: (float) end-of-year storage carryover capacity
    :param k_count: (int) number of piecewise links
    :returns x: (numpy.ndarray) array of storage values 
    :returns y: (numpy.ndarray) array of penalty values as function of storage values 
    """
    # determine COSVF coefficients based on pmin and pmax and reservoir capacity
    a = (pmin - pmax) / (2 * eop_max)
    b = pmax
    c = -1 * (eop_max * (pmin + pmax)) / 2

    # build COSVF curve with penalties based on end-of-year storage series
    x = np.linspace(eop_min, eop_max, k_count+1)
    y = a * x**2 + b * x + c

    return x, y, a, b, c


  def cosvf_marginal_piecewise(self, x, y):
    """
    Calculate slope (cost) and breakpoints (k) for the fitted piecewise quadratic COSVF

    :param x: (numpy.ndarray) array of storage values
    :param y: (numpy.ndarray) array of penalty values for x array of storage values
    :returns r_b: (list) storage breakpoints
    :returns r_k: (list) and corresponding slopes (marginal values)
    """
    breaks, slopes = np.zeros(len(x)-1), np.zeros(len(x)-1)
    for i in range(0, len(x)-1):
      dx = x[i+1] - x[i]
      slope = (y[i+1] - y[i]) / dx
      # k=0 upper bound includes dead pool (0 to x[1]); k>0 is just segment width
      width = x[i+1] if i == 0 else dx
      breaks[i], slopes[i] = width, slope
    return breaks, slopes


  def cosvf_construct_piecewise_penalties(self, r):
    """
    Create piecewise costs for penalties on end-of-year storage for
    rtype1 (quadratic) and rtype2 (linear) COSVF penalties.
    
    :param r: (str) reservoir id (e.g. "SR_DNP")
    :returns r_b: (list) storage breakpoints
    :returns r_k: (list) and corresponding slopes (marginal values)
    """
    # Pmin Pmax for r-type 1 (quadratic COSVF)
    if self.r_dict[r]['type']==1:
      # construct COSVF from params
      cosvfx, cosvfy, a, b, c = self.cosvf_fit_from_params(
        pmin=self.pcosvf[self.r_dict[r]['cosvf_param_index'][0]], 
        pmax=self.pcosvf[self.r_dict[r]['cosvf_param_index'][1]],
        eop_min=self.r_dict[r]['lb'],
        eop_max=self.r_dict[r]['ub'],
        k_count=self.r_dict[r]['k_count'])
      
      # get piecewise storage breakpoints and penalty slopes
      r_b, r_k = self.cosvf_marginal_piecewise(cosvfx, cosvfy)

    # linear penalty for r-type 2 (linear COSVF)
    elif self.r_dict[r]['type']==2:
      r_b = [self.r_dict[r]['eop_init'],  self.r_dict[r]['ub'] - self.r_dict[r]['eop_init']]
      r_k = [self.pcosvf[self.r_dict[r]['cosvf_param_index']], 0]

    return r_b, r_k


  def assign_cosvf_penalties(self):
    """
    Assign the COSVF values to links on the model. 

    :returns: nothing, but modifies COSVF CALVIN model object
    """
    for r in self.r_dict:

      if self.r_dict[r]['type']>=1:
        # get penalty link costs and breakpoints
        links_b, links_k  = self.cosvf_construct_piecewise_penalties(r)

        # assign piecewise COSVF to calvin model reservoir links
        for k in range(self.r_dict[r]['k_count']):
          self.model.c[('{}.{}-09-30'.format(r,self.wy_start), 'FINAL', k)] = links_k[k]
          self.model.u[('{}.{}-09-30'.format(r,self.wy_start), 'FINAL', k)] = links_b[k]


  def cosvf_update_initial_storage(self, eop):
    """
    Update initial storages in COSVF annual mode

    :param eop: (dict) dictionary of reservoir nodes with the end of year storage
      from the previous water year's solution
    :returns: nothing, but modifies CALVIN model object
    """
    if eop is not None:
      for r, key in self._initial_storage_keys.items():
        v = eop[r]
        self.model.l[key] = v
        self.model.u[key] = v


  def cosvf_update_inflows(self, wy):
    """
    Update link inflows to reflect the current water year under analysis.

    :param wy: (int) current water year under evaluation.
    :returns: nothing, but modifies CALVIN model object
    """
    row = self._inflow_matrix[wy - self.wy_start]
    for idx, (ki, kj) in enumerate(self._inflow_keys):
      v = row[idx]
      self.model.l[(ki, kj, 0)] = v
      self.model.u[(ki, kj, 0)] = v


  def cosvf_update_variable_bounds(self, wy):
    """
    Update link lower/upper bounds to reflect the current water year under analysis.

    :param wy: (int) current water year under evaluation.
    :returns: nothing, but modifies CALVIN model object
    """
    for (ki, kj, k, lb, ub) in self._vc_by_year[wy]:
      self.model.l[(ki, kj, k)] = lb
      self.model.u[(ki, kj, k)] = ub


#####################################################
### DEAP evolutionary algorithm for COSVF search ###
#####################################################
def cosvf_ea_main(toolbox, n_gen, mu, pwd, cxpb=1, mutpb=1, seed=None, log_name='calvin-cosvf-ea', checkpoint=None):
  """
  Main evolutionary algorithm using NSGA-III selection.

  :param toolbox: (object) the DEAP toolbox constructed using ``cosvf_ea_toolbox``
  :param n_gen: (object) number of evolutionary generations to conduct (stopping criteria)
  :param mu: (int) number of individuals in the evolutionary population
  :param pwd: (path) directory to save evolutionary results and checkpoints
  :param cxpb: (float) [0,1] probability of mating two individuals (consecutive pairs in pop)
  :param mutpb: (float) [0,1] probability of mutating an individual
  :param seed: (int) random seed (will assign random integer b/t 1 and 100 if not specified)
  :param log_name: (string) global logger name to use, log file will save to ``pwd``
  :param checkpoint: (path) checkpoint file of previous EA to continue running
  :returns: nothing, but outputs evolutionary results to CSV and a pickled checkpoint
  """ 
  import time

  if checkpoint:
    with open(os.path.join(pwd,checkpoint), "rb") as cp_file:
        cp = pickle.load(cp_file)
    pop = cp["population"]
    start_gen = cp["generation"]
    logbook = cp["logbook"]
    seed = int(''.join(filter(str.isdigit, checkpoint)))
    random.setstate(cp["random_state"])
  else:
    start_gen = 0
    seed = random.randint(1,100) if seed is None else seed
    random.seed(seed)
    # initialize logbook
    logbook = tools.Logbook()
    # initialize population
    pop = toolbox.population(n=mu)

  # set up logging code
  if not os.path.isdir(pwd): os.makedirs(pwd)
  log = setup_logger(log_name=log_name+'_seed'+str(seed), savedir=pwd)
  log.info('------Evolutionary search for COSVF---------')
  log.info('Pop={} | Gen={} | Seed={}'.format(mu, n_gen, seed))

  # Initialize statistics objects
  pop_hist = tools.Statistics()
  pop_hist.register("pop", copy.deepcopy)
  fitness_hist = tools.Statistics(lambda ind: ind.fitness.values)
  fitness_hist.register("fitnesses", copy.deepcopy)
  fit_stats = tools.Statistics(lambda ind: ind.fitness.values)
  fit_stats.register("avg", np.mean, axis=0)
  fit_stats.register("std", np.std, axis=0)
  fit_stats.register("min", np.min, axis=0)
  fit_stats.register("max", np.max, axis=0)

  # Console table header — printed once before the loop
  _hdr = '{:>5}  {:>6}  {:>14}  {:>14}  {:>14}  {:>8}'
  _row = '{:>5}  {:>6}  {:>14}  {:>14}  {:>14}  {:>8}'
  print(_hdr.format('gen', 'evals', 'best(f1,f2,f3)', 'avg(f1,f2,f3)', 'std(f1,f2,f3)', 'gen_time'), flush=True)

  # Optional tqdm progress bar on the generation loop
  try:
    from tqdm import tqdm as _tqdm
    _gen_iter = _tqdm(
      range(start_gen, n_gen + 1),
      total=n_gen + 1 - start_gen,
      unit='gen',
      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} gen [{elapsed}<{remaining}]',
      leave=True,
    )
  except ImportError:
    log.warning('tqdm not installed; no generation progress bar')
    _gen_iter = range(start_gen, n_gen + 1)

  ### Evolutionary loop ###
  _wall_start = time.time()
  for gen in _gen_iter:
    _gen_t0 = time.time()

    if gen==start_gen:
      # Evaluate the individuals with an invalid fitness
      invalid_ind = [ind for ind in pop if not ind.fitness.valid]
      fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
      for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # offspring generated from crossover and mutation
    offspring = algorithms.varAnd(pop, toolbox, cxpb=cxpb, mutpb=mutpb)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
      ind.fitness.values = fit

    # Select the next generation population from parents and offspring
    pop = toolbox.select(pop + offspring, mu)

    # Compile statistics about the new population
    logbook.record(gen=gen, evals=len(invalid_ind), 
             **pop_hist.compile(pop), 
             **fitness_hist.compile(pop),
             **fit_stats.compile(pop))

    _gen_elapsed = time.time() - _gen_t0
    _mins, _secs = divmod(int(_gen_elapsed), 60)
    _gen_time_str = '{:d}m{:02d}s'.format(_mins, _secs)

    _best = logbook.select('min')[-1]
    _avg  = logbook.select('avg')[-1]
    _std  = logbook.select('std')[-1]
    _best_str = '({})'.format(','.join('{:.1f}'.format(f) for f in _best))
    _avg_str  = '({})'.format(','.join('{:.1f}'.format(f) for f in _avg))
    _std_str  = '({})'.format(','.join('{:.1f}'.format(f) for f in _std))

    # Single compact summary line per generation (goes to console + log file)
    _summary = _row.format(gen, len(invalid_ind), _best_str, _avg_str, _std_str, _gen_time_str)
    log.info(_summary)

    if hasattr(_gen_iter, 'set_postfix'):
      _gen_iter.set_postfix(
        best_f1='{:.1f}'.format(_best[0]),
        best_f2='{:.2f}'.format(_best[1]),
        evals=len(invalid_ind),
      )

    # checkpoint every generation
    cp = dict(population=pop, generation=gen, logbook=logbook, random_state=random.getstate())
    with open(os.path.join(pwd,"cosvf-ea-chkpnt-{}.pickle".format(seed)), "wb") as cp_file:
      pickle.dump(cp, cp_file)

  _total_elapsed = time.time() - _wall_start
  _h, _rem = divmod(int(_total_elapsed), 3600)
  _m, _s   = divmod(_rem, 60)
  log.info('EA complete: total wall time {:d}h{:02d}m{:02d}s'.format(_h, _m, _s))

  # save out logbook to csv
  logbook_to_csv(logbook, pwd, seed)


def cosvf_ea_toolbox(cosvf_evaluate, nrtype, mu, nobj=3, cx_eta=10., mut_eta=40., mutind_pb=0.5):
  """
  Create a DEAP toolbox with the NSGA-III selection evolutionary algorithm.

  :param cosvf_evaluate: (func) this function, which must be defined in the "main" run file, constructs
    a COSVF CALVIN model object, taking COSVF params as the argument for the model solve
  :param nrtype: (list) [(int), (int)] a list with number of type 1 (quadratic) COSVF reservoirs 
    as the first entry and number of type 2 (linear) COSVF reservoirs as the second entry
  :param mu: (int) number of individuals in the population
  :param nobj: (int) number of objectives
  :param cx_eta: (float) Likeness degree of the simulated binary bounded crossover. 
    High eta --> children close to parents; Low eta --> children far from parents
  :param mut_eta: (float) Likeness degree of the polynomial bounded mutation. 
  :param mutind_pb: (float) [0,1] probability of mutating a parameter within a given individual.
  :returns: a DEAP toolbox for the evolutionary search
  """
  # individuals
  creator.create("FitnessMin", Fitness, weights=(-1.0,) * nobj)
  creator.create("Individual", list, fitness=creator.FitnessMin)

  # toolbox
  toolbox = Toolbox()

  # population
  n_param=(nrtype[0]*2)+nrtype[1]
  toolbox.register("attr_pminmax", random.uniform, -6.0e2, -1.)
  toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_pminmax, n=n_param)
  toolbox.register("population", tools.initRepeat, list, toolbox.individual)

  # mating
  toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=-1.5e3, up=0., eta=cx_eta)
  # mutation
  toolbox.register("mutate", tools.mutPolynomialBounded, low=-1.5e3, up=0., eta=mut_eta, indpb=mutind_pb)

  # bound check
  toolbox.decorate("population", cosvf_check_bounds(nrtype[1],init=True))
  toolbox.decorate("mate", cosvf_check_bounds(nrtype[1],init=False))
  toolbox.decorate("mutate", cosvf_check_bounds(nrtype[1],init=False))

  # evaluation
  toolbox.register("evaluate", cosvf_evaluate)

  # selection
  pdiv = pdiv_from_mu(mu,nobj=nobj)
  ref_points = tools.uniform_reference_points(nobj=nobj, p=pdiv)
  toolbox.register("select", tools.selNSGA3WithMemory(ref_points))

  return toolbox


def cosvf_check_bounds(rtype1_start_idx, init):
  """
  Check bounds of indiviudal's COSVF Pmin and Pmax array during evolution.

  Two checks: 
    - Minimum penalty is zero
    - Pmax cannot be greater than Pmin

  :param rtype1_start_idx: (int) position on individual parameter list at which rtype1 begin
  :returns decorator: (func) a decorator function that is applied after individual mating or mutation
  """
  def decorator(func):
    def wrapper(*args, **kargs):
      population = func(*args, **kargs)
      for ind in population:
        # check pmin bounds (greater than zero)
        for idx in range(0, len(ind)):
          if ind[idx] > 0:
            ind[idx] = -1*random.uniform(1, 10)
        # check pmax > pmin violation for r_type1 (pminmax COSVF)
        for pmax in range(rtype1_start_idx+1, len(ind), 2):
          if ind[pmax] > ind[pmax-1]:
            ind[pmax] = max(ind[pmax-1]+(ind[pmax-1]*random.uniform(0.05, 1)), -1.99e3)
        if init == True:
          for idx in range(0, rtype1_start_idx+1):
            ind[idx] = -1.0
      return population
    return wrapper
  return decorator


def mu_from_pdiv(pdiv, nobj=3):
  """
  Get population count based on divisions per objective for NSGA-III
  """
  h = int(math.factorial(nobj + pdiv - 1) / (math.factorial(pdiv) * math.factorial(nobj - 1)))
  mu = int(h + (4 - h % 4))
  return mu


def pdiv_from_mu(mu, nobj=3):
  """
  Get divisions per objective for NSGA-III from a population count.
  """
  div = Symbol('n', integer=True)
  expr = (factorial(nobj+div-1) / (factorial(div) * factorial(nobj-1))) - mu
  pdiv = int(nsolve(expr,1))
  return pdiv


def logbook_to_csv(logbook, pwd, seed):
  """
  Convert the logbook of evolutionary history to a CSV file.

  :param logbook: (object) DEAP logbook of evolutionary history
  :param pwd: (path) directory to save evolutionary results and checkpoints
  :param seed: (int) random seed used in EA generations
  :returns: nothing, but outputs CSV file ``cosvf-ea-history-seed[number].csv``
  """
  # dictionary of surface and groundwater reservoir nodes
  with open(os.path.join(pwd, 'r-dict.json')) as f: 
      r_dict = json.load(f)
  rtype1_list = list({key: value for key, value in r_dict.items() if value['type'] == 1}.keys())
  rtype2_list = list({key: value for key, value in r_dict.items() if value['type'] == 2}.keys())
  ind_list = list(np.array(range(len(logbook.select('pop')[0]))) + 1)
  param = ['pmin', 'pmax']

  pos_df = pd.DataFrame()
  cost_df = pd.DataFrame()

  for iteration in np.arange(0,len(logbook.select('pop'))):
    # get linear penalties
    p_df = pd.DataFrame(
      {'gen': np.repeat(iteration, len(rtype2_list) * len(ind_list)),
      'ind': np.repeat(ind_list, len(rtype2_list)),
      'r': list(rtype2_list) * len(ind_list),
      'param': ['p'] * len(rtype2_list) * len(ind_list)})
    pos = pd.melt(pd.DataFrame(np.array(logbook.select('pop')[iteration])).T[0:len(rtype2_list)])['value']
    pos_df = pd.concat([pos_df, pd.concat([p_df, pos], axis=1)], ignore_index=True)

    p_df = pd.DataFrame(
      {'gen': np.repeat(iteration, len(rtype1_list) * len(param) * len(ind_list)),
      'ind': np.repeat(ind_list, len(rtype1_list) * len(param)),
      'r': list(np.repeat(rtype1_list, len(param))) * len(ind_list),
      'param': param * len(rtype1_list) * len(ind_list)})
    pos = pd.melt(pd.DataFrame(np.array(logbook.select('pop')[iteration])).T[-len(rtype1_list)*2:])['value']
    pos_df = pd.concat([pos_df, pd.concat([p_df, pos], axis=1)], ignore_index=True)

    c_df = pd.DataFrame(
      {'gen': iteration, 'ind': ind_list,
      'f1': np.array(logbook.select('fitnesses')[iteration]).T[0],
      'f2':np.array(logbook.select('fitnesses')[iteration]).T[1],
      'f3':np.array(logbook.select('fitnesses')[iteration]).T[2]})
    cost_df = pd.concat([cost_df, c_df], ignore_index=True)

  df = pos_df.merge(cost_df, on=['gen','ind'])
  df.to_csv(os.path.join(pwd, 'cosvf-ea-history-seed{}.csv'.format(seed)),index=False)

