import os, shutil, re, itertools, json, pickle
import argparse
import logging
import numpy as np
import pandas as pd
import math, copy, random
from sympy import Symbol, factorial, nsolve
from deap import algorithms, creator, tools
from deap.base import Fitness
from deap.base import Toolbox
from .calvin import *
from .postprocessor import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

###############################################################################
### Limited foresight Carryover storage value function (COSVF) CALVIN model ###
###############################################################################
class COSVF(CALVIN):
  
  def __init__(self, pwd, log_name="calvin-cosvf"):
    """
    Instantiate COSVF model as a child class of ``calvin.CALVIN`` for annual COSVF optimization.

    :param pwd: (string) path to directory containting COSVF links and other required input files:
      1. ``links.csv``: network for the first water year in the period of analysis.
          CSV file with column headers: ``i,j,k,cost,amplitude,lower_bound,upper_bound``
      2. ``cosvf-params.csv``: a csv containing a table of :math:`P_{min}` and :math:`P_{max}` for 
          quadratic carryover penalty curves on surface water reservoirs and :math:`P_{GW}` 
          for linear penalty on groundwater reservoirs.
          CSV file with column headers: ``r,param,value``
      3.  ``r-dict.json``: dictionary of reservoirs in the network with penalty properties. 
          Type 2 (linear penalty) reservoirs must be ordered prior to the Type 1 (quadratic) reservoirs.
          This is a limitation of imposed by the evolutionary algorithm search of the parameters.
          For each reservoir with an EOP penalty, an index attribute ``cosvf_param_index`` points 
          to the row-index (pythonic zero-indexed) of the list of reservoirs in ``cosvf-params.csv``.
          Dictionary structure:
            .. code-block:: JSON
              {"<reservoir id (e.g. SR_DNP)>":
                {
                  "eop_init": "(float) initial (October 1) storage level",
                  "lb": "(float) minimum (end-of-September) storage level",
                  "ub": "(float) maximum (end-of-September) carryover capactiy",
                  "type": "(int) 0:none; 1:quadratic; or 2:linear",
                  "cosvf_param_index": "(list) index to cosvf_params.csv row (pythonic zero-indexed)",
                  "k_count": "(int) number of piecewise links to fit for quadratic COSVF",
                }
              }
      4.  ``inflows.csv``: external inflows for every monthly time step over the period of analysis to run. 
          CSV file with column headers ``date, j, flow_taf``
      5.  ``variable-constraints.csv``: links that have variable year-to-year upper and/or lower bounds.
          CSV file with column headers ``date,i,j,k,lower_bound,upper_bound``
    :param log_name: (string) name for the global logger. Log file is written to the specified ``pwd`` path.
    :returns: COSVF CALVIN model object
    """
    # set working directory
    self.pwd = pwd

    # if COSVF links file does not exist in pwd, copy the default files to the pwd
    linksfile = os.path.join(self.pwd , 'links.csv')
    if not os.path.isfile(linksfile):
      if not os.path.isdir(self.pwd): os.makedirs(self.pwd)
      src, dst = os.path.join(BASE_DIR, "data", "cosvf-default"), self.pwd
      names = os.listdir(src)
      for name in names:
        if name.endswith('py') or name.endswith('md') or name.endswith('default.csv'):
          continue
        else:
          shutil.copy2(os.path.join(src, name), os.path.join(dst, name))

    # set up logging code
    self.log = setup_logger(log_name, savedir=pwd)

    # load links
    self.df = pd.read_csv(os.path.join(self.pwd , 'links.csv'))
    
    # dictionary of surface and groundwater reservoir nodes
    with open(os.path.join(self.pwd , 'r-dict.json')) as f: 
      self.r_dict = json.load(f)

    # COSVF parameter array that matches the order of reservoirs in the r_dict
    self.pcosvf = np.loadtxt(os.path.join(self.pwd , 'cosvf-params.csv'),
                            delimiter=',', skiprows=1, usecols=2).tolist()

    # inflows for entire period of analysis
    inflows = pd.read_csv(os.path.join(self.pwd , 'inflows.csv'), index_col=0, parse_dates=True)
    self.inflows = inflows.pivot(columns='j', values='flow_taf').rename_axis(None, axis=1)
    
    # wy start and end inferred from inflows
    self.wy_start = int(min(self.inflows.index.year)) + 1
    self.wy_end = int(max(self.inflows.index.year))
    
    # get a list of unique inflow points to loop through later
    self.inflow_terminals = inflows.j.unique()

    # load time-varying upper and lower bounds for entire period of analysis
    self.variable_constraints = pd.read_csv(os.path.join(self.pwd , 'variable-constraints.csv'),
      index_col=0, parse_dates=True)

    # construct placeholder cosvf cost links for the reservoirs
    self.create_cosvf_links()

    # construct the link index for the model dataframe
    self.df['link'] = self.df.i.map(str) + '_' + self.df.j.map(str) + '_' + self.df.k.map(str)
    self.df.set_index('link', inplace=True)

    # SR stats for hydropower checks
    SR_stats = pd.read_csv(os.path.join(BASE_DIR,'data','SR_stats.csv'), index_col=0).to_dict()
    self.min_storage = SR_stats['min']
    self.max_storage = SR_stats['max']

    # a few network fixes to make things work
    super().add_ag_region_sinks()
    super().fix_hydropower_lbs()

    # lists for unique nodes and links
    self.nodes = pd.unique(self.df[['i','j']].values.ravel()).tolist()
    self.links = list(zip(self.df.i,self.df.j,self.df.k))

    # make sure things aren't broken
    super().networkcheck()


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


  def cosvf_solve(self, solver='glpk', nproc=1, resultdir=None, pcosvf=None):
    """
    Solve COSVF CALVIN model for full period of analysis

    :param solver: (string) solver name. glpk, cplex, cbc, gurobi.
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
    :returns : 
    """
    # declare solver
    from pyomo.opt import SolverFactory
    opt = SolverFactory(solver)
    if nproc > 1 and solver != 'glpk':
      opt.options['threads'] = nproc

    # overwrite pcosvf (used for evolutionary mode)
    if pcosvf is not None: self.pcosvf = pcosvf

    # assign COSVF penalties to r links 
    self.assign_cosvf_penalties()
    
    # output final model links file (for reference)
    model_df = super().model_to_dataframe()
    model_df.to_csv(os.path.join(self.pwd,'links-pyomo-model-reference.csv'))

    # initialize fitness 1 (value is cumulative over period-of-analysis)
    f1 = 0

    # calculate avg COSVF penalty
    f3 = np.mean(np.array(self.pcosvf)) * -1

    # loop through years in sequence
    for wy in range(self.wy_start, self.wy_end + 1):
      # number of years so far evaluated in the annual sequence
      years = wy - self.wy_start + 1

      # update storages, inflows, and (variable) constraints for the wy
      if wy != self.wy_start:
        eop = {}
        for r in self.r_dict:
          eop[r] = 0
          for k in range(self.r_dict[r]['k_count']):
            eop[r] += self.model.X[('{}.{}-09-30'.format(r,self.wy_start), 'FINAL', k)].value
        self.cosvf_update_initial_storage(eop=eop)

      # update inflows for current water year
      self.cosvf_update_inflows(wy=wy)

      # update constraints which vary across water years
      if self.variable_constraints is not None: self.cosvf_update_variable_bounds(wy=wy)

      # solve model
      self.log.debug('-----Solving Pyomo Model (wy=%d)' % wy)
      self.results = opt.solve(self.model, keepfiles=False)
      
      # check solver solution status
      if self.results.solver.termination_condition == TerminationCondition.optimal:
        
        # load solution to model
        self.model.solutions.load_from(self.results)

        # join flows and costs for fitness calcs
        model_df = super().model_to_dataframe()

        # calc fitness
        short_costs, op_costs = self.compute_network_costs(model_df)
        f1 += (short_costs + op_costs)
        f2 = self.compute_gw_overdraft(model_df)

        self.log.debug('Costs \\$M/yr=%.1f; GW O.D. MAF/yr=%.1f' % (f1/1e3/years, f2/1e3/years))

        # postprocessing and saving
        if resultdir is not None: 
          postprocess(self.df, self.model, resultdir=resultdir, annual=True, year=wy)

      else:
        # Something else is wrong
        self.log.info('Solver issue! Fitness values set to infinite')
        return(np.inf, np.inf, np.inf)

    # normalize fitness values by period of record length
    f1 = f1 / 1e3 / years
    f2 = f2 / 1e3 / years

    return f1, f2, f3


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
    # drop storage persuasion penalties
    cost_links = cost_links.drop(cost_links[(cost_links['i'].str.contains('SR')) &
                                        (cost_links['i'].str.contains('SR'))].index)                                   
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
    :param wy: (int) current water year being evaluated in the annual sequence 
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
      dist = (x[i+1]-x[i]) if i>0 else x[i] + (x[i+1]-x[i])
      slope = (y[i+1]-y[i]) / dist
      breaks[i], slopes[i] = dist, slope 
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
        k_count=15)
      
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
    # update initial storage condition
    if eop is not None:
      for r in eop:
        self.model.l[('INITIAL', '{}.{}-10-31'.format(r,self.wy_start-1), 0)] = eop[r]
        self.model.u[('INITIAL', '{}.{}-10-31'.format(r,self.wy_start-1), 0)] = eop[r]


  def cosvf_update_inflows(self, wy):
    """
    Update link inflows to reflect the current water year under analysis.
    
    :param wy: (int) current water year under evaluation.
    :returns: nothing, but modifies CALVIN model object
    """ 
    # replace inflows for current wy in sequence
    offset = wy - self.wy_start
    dates = pd.date_range('{}-10-31'.format(wy - 1), '{}-09-30'.format(wy), freq='ME')
    for t in self.inflow_terminals:
      for date in dates:
        inflow = self.inflows.loc[date, t]
        fdate = str((date - pd.DateOffset(years=offset)).date())
        self.model.l[('INFLOW.{}'.format(fdate), '{}.{}'.format(t, fdate), 0)] = inflow
        self.model.u[('INFLOW.{}'.format(fdate), '{}.{}'.format(t, fdate), 0)] = inflow


  def cosvf_update_variable_bounds(self, wy):
    """
    Update link lower/upper bounds to reflect the current water year under analysis.
    
    :param wy: (int) current water year under evaluation.
    :returns: nothing, but modifies CALVIN model object
    """
    offset = wy - self.wy_start 
    dates = pd.date_range('{}-10-31'.format(wy - 1), '{}-09-30'.format(wy), freq='ME')
    variable_constraints_wy = self.variable_constraints.loc[dates]
    for idx, row in variable_constraints_wy.iterrows():
      i_node, i_date = row.i.split('.')[0], row.i.split('.')[1]
      i_date = str((pd.to_datetime(i_date, format='%Y-%m-%d') - pd.DateOffset(years=offset)).date())
      i = i_node+'.'+i_date
      try:
        j_node, j_date = row.j.split('.')[0], row.j.split('.')[1]
        j_date = str((pd.to_datetime(j_date, format='%Y-%m-%d') - pd.DateOffset(years=offset)).date())
        j = j_node+'.'+j_date
      except:
        j = 'FINAL'
      k = float(row.k)
      l = row.lower_bound
      u = row.upper_bound
      self.model.l[(i,j,k)] = l
      self.model.u[(i,j,k)] = u


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

  ### Evolutionary loop ###
  for gen in range(start_gen, n_gen+1):

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
    log.info('----------------Generation = {}-------------'.format(gen))
    log.info('min(f)={}'.format(["%.2f" % f for f in logbook.select('min')[-1]]))
    log.info('max(f)={}'.format(["%.2f" % f for f in logbook.select('max')[-1]]))
    log.info('avg(f)={}'.format(["%.2f" % f for f in logbook.select('avg')[-1]]))
    log.info('std(f)={}'.format(["%.2f" % f for f in logbook.select('std')[-1]]))

    # checkpoint ea every 5 generations
    if gen % 5 == 0:
      cp = dict(population=pop, generation=gen, logbook=logbook, random_state=random.getstate())
      with open(os.path.join(pwd,"cosvf-ea-chkpnt-{}.pickle".format(seed)), "wb") as cp_file:
        pickle.dump(cp, cp_file)

  # checkpoint the final iteration
  cp = dict(population=pop, generation=gen, logbook=logbook, random_state=random.getstate())
  with open(os.path.join(pwd,"cosvf-ea-chkpnt-{}.pickle".format(seed)), "wb") as cp_file:
    pickle.dump(cp, cp_file)

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
