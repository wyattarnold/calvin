Overview
=========

CALVIN is a network flow optimization model of California's water supply system. It uses
`Pyomo <https://www.pyomo.org/>`_ to formulate linear programs over a node-link network and
solves them with LP/MIP solvers (HiGHS, CBC, Gurobi, CPLEX).


Installation
-------------

Clone the repo and install into your environment. The install step is required so that
``calvin`` is importable from any working directory (including ``scripts/``).

**pip:**

.. code-block:: bash

   git clone https://github.com/wyattarnold/calvin.git
   cd calvin
   pip install -e ".[solver]"

**conda:**

.. code-block:: bash

   git clone https://github.com/wyattarnold/calvin.git
   cd calvin
   conda create -n calvin python=3.11
   conda activate calvin
   pip install -e ".[solver]"

``pip`` works inside conda environments and will install all dependencies. The ``-e``
flag installs the package in editable mode so that source changes take effect immediately.

To run the web app locally, use the ``app`` extra instead of (or in addition to) ``solver``:

.. code-block:: bash

   pip install -e ".[app]"


The model can be run in three modes:

1. **Perfect foresight** — a single large LP over the full time horizon (e.g. 82 water years),
   solved once.
2. **Annual (constraint-based)** — a sequence of single-year LPs where end-of-period
   storage is managed by imposing minimum storage constraints as a fraction of reservoir
   capacity. No economic penalties are used.
3. **Annual (COSVF + evolutionary)** — a sequence of single-year LPs connected by
   Carryover Storage Value Functions (COSVFs) that penalize end-of-year reservoir storage
   to approximate the value of water carried into the next year. Penalty parameters are
   optimized via an evolutionary algorithm.


Perfect Foresight Mode
-----------------------

In perfect foresight mode, the ``CALVIN`` class loads a single links CSV containing the
full time-expanded network and solves it directly:

.. code-block:: python

   from calvin import CALVIN, postprocess

   calvin = CALVIN('links82yr.csv')
   calvin.create_pyomo_model(debug_mode=True, debug_cost=2e10)
   calvin.solve_pyomo_model(solver='highs', nproc=1, debug_mode=True)

   calvin.create_pyomo_model(debug_mode=False)
   calvin.solve_pyomo_model(solver='highs', nproc=1, debug_mode=False)

   postprocess(calvin.df, calvin.model, resultdir='results')


Annual Mode (Constraint-Based)
-------------------------------

The simplest limited-foresight approach solves one water year at a time without economic
storage penalties. Instead, the ``eop_constraint_multiplier`` method sets the lower bound on
end-of-September reservoir storage to a fraction of each reservoir's capacity. This prevents
the optimizer from completely emptying reservoirs within a single year.

The method uses ``SR_stats.csv`` (loaded automatically by CALVIN) which contains
``min`` and ``max`` storage for each surface reservoir. For a given fraction *x*, the
end-of-period lower bound for reservoir *k* is set to:

.. math::

   LB_k = S_{min,k} + (S_{max,k} - S_{min,k}) \cdot x

where :math:`S_{min,k}` and :math:`S_{max,k}` are the minimum and maximum storage from
``SR_stats.csv``.

The annual loop requires one links CSV file **per water year** (e.g. exported from
``calvin.network``). End-of-period storage from each year is passed as initial conditions
to the next via the ``ic`` parameter:

.. code-block:: python

   from calvin import CALVIN, postprocess

   eop = None

   for wy in range(1922, 2004):
       print(f'\nNow running WY {wy}')

       calvin = CALVIN(f'calvin/data/annual/linksWY{wy}.csv', ic=eop)
       calvin.eop_constraint_multiplier(0.1)

       calvin.create_pyomo_model(debug_mode=True, debug_cost=2e8)
       calvin.solve_pyomo_model(solver='highs', nproc=1, debug_mode=True, maxiter=15)

       calvin.create_pyomo_model(debug_mode=False)
       calvin.solve_pyomo_model(solver='highs', nproc=1, debug_mode=False)

       # postprocess appends to per-year result directories; returns EOP storage for next year
       eop = postprocess(calvin.df, calvin.model,
                         resultdir=f'results/annual/WY{wy}', annual=True)

.. note::

   The constraint fraction (here ``0.1``, i.e. 10% of capacity) is a tunable parameter.
   Lower values give the optimizer more freedom but risk over-drafting storage; higher
   values are more conservative.


Combining Annual Results
~~~~~~~~~~~~~~~~~~~~~~~~~

After the annual loop completes, use :func:`calvin.postprocessor.combine_annual_results`
to concatenate the per-year CSV files into single timeseries files:

.. code-block:: python

   from calvin import combine_annual_results

   combine_annual_results(
       years=range(1922, 2004),
       annual_dir='results/annual',
       output_dir='results',
   )

This reads ``results/annual/WY{year}/*.csv`` for each year and writes concatenated files
to ``results/``.


Annual Mode (COSVF + Evolutionary)
------------------------------------

The COSVF approach also solves the network one water year at a time, but replaces the
simple storage constraints with economic penalty functions on end-of-period storage. These
**Carryover Storage Value Functions** represent the marginal value of storing water for
future use.

Two penalty types are supported:

- **Type 1 (quadratic)**: for surface reservoirs — defined by :math:`P_{min}` and
  :math:`P_{max}` parameters that shape a quadratic penalty curve between minimum
  operating storage and full carryover capacity. The curve is linearized into piecewise
  segments for the LP.
- **Type 2 (linear)**: for groundwater reservoirs — a single marginal penalty :math:`P_{GW}`
  applied to storage below the initial level.


Evolutionary Optimization of COSVF Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The penalty parameters are not known a priori. The ``cosvfea`` module uses the
**NSGA-III** multi-objective evolutionary algorithm (via `DEAP <https://deap.readthedocs.io/>`_)
to search for optimal penalty values. The three objective functions minimized are:

1. **Shortage + operational costs** (\\$/year) — total annualized cost across all demand and
   operational links.
2. **Groundwater overdraft** (MAF/year) — net depletion across all groundwater basins.
3. **Mean penalty magnitude** — regularization to avoid unnecessarily large penalties.

Each candidate solution (individual) is a vector of penalty parameters for all reservoirs.
The EA evaluates each individual by running the full annual COSVF sequence and computing
the three fitness values. The search is designed to run in parallel using ``mpi4py``.

.. code-block:: python

   # main-cosvfea.py (simplified)
   from calvin import cosvfea

   def cosvf_evaluate(pcosvf):
       calvin = cosvfea.COSVF(pwd='./my-models/calvin-cosvf')
       calvin.create_pyomo_model(debug_mode=True)
       return calvin.cosvf_solve(solver='cbc', nproc=1, pcosvf=pcosvf)

   toolbox = cosvfea.cosvf_ea_toolbox(
       cosvf_evaluate=cosvf_evaluate,
       nrtype=[26, 32],  # 26 quadratic, 32 linear reservoirs
       mu=95
   )

Running with MPI:

.. code-block:: bash

   mpirun -n <ncpus> python main-cosvfea.py


COSVF Ending-Storage Constraint
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``prepare_cosvf`` calls ``build_matrix()`` with ``constrain_ending='all'``, which
sets ``lb = ub = final_val`` on every storage node's ``→ FINAL`` link.  This
fixes ending storage in the **template water-year LP** used to build the
single-step network structure. However, in an actual COSVF run the ending-storage
constraint is **not** enforced this way.  The COSVF solver replaces the ``→ FINAL``
link bounds with the piecewise-linear penalty curve for each reservoir: ending
storage is a free decision variable penalised by the COSVF objective term, not pinned
to a fixed target.


Preparing COSVF Input Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before running a COSVF model, generate the required input files using
:func:`calvin.network.prepare.prepare_cosvf`.  This reads the
`calvin-network-data <https://github.com/ucd-cws/calvin-network-data>`_ repository
directly (no prior perfect-foresight run required) and writes five files into the
specified output directory:

.. code-block:: bash

   python -m calvin.network.cli prepare-cosvf \
       --data ../calvin-network-data/data \
       --output ./my-models/calvin-cosvf

Or equivalently from Python:

.. code-block:: python

   from calvin.network import prepare_cosvf
   prepare_cosvf(data_path='../calvin-network-data/data',
                 output_dir='./my-models/calvin-cosvf')

The generated files are:

``links.csv``
   Network for a single water year (WY 1922). This serves as the template LP that is
   re-solved for each year in sequence with updated inflows and inital storage conditions.

   Schema: ``i,j,k,cost,amplitude,lower_bound,upper_bound``

``cosvf-params.csv``
   Penalty parameters for each reservoir. Contains columns ``r,param,value`` where:

   - For **Type 2** (groundwater) reservoirs: a single row with ``param=p`` and the linear
     penalty value.
   - For **Type 1** (surface) reservoirs: two rows per reservoir with ``param=pmin`` and
     ``param=pmax`` defining the quadratic curve endpoints.

   .. note::

      The default values are placeholders (e.g. ``-100.0`` for all groundwater).
      They are meant to be replaced by the evolutionary optimization.

``r-dict.json``
   Dictionary of reservoirs keyed by node ID (e.g. ``SR_SHA``, ``GW_01``). Each entry
   defines:

   - ``eop_init``: target end-of-period storage level (TAF).  Prefers
     ``endingstorage`` from the network data; falls back to ``initialstorage``
     if no ending storage is defined.
   - ``lb``: minimum end-of-September storage (TAF)
   - ``ub``: maximum carryover capacity (TAF)
   - ``type``: ``0`` (no penalty), ``1`` (quadratic), or ``2`` (linear)
   - ``cosvf_param_index``: row index into ``cosvf-params.csv`` (zero-indexed)
   - ``k_count``: number of piecewise segments for the penalty curve

   .. important::

      **All GW nodes must precede all SR nodes** in ``r-dict.json``.  Within each
      group the ordering is: type-2 (sorted alphabetically), then type-0 (sorted);
      followed by type-1 (sorted), then type-0 (sorted).  The EA parameter vector
      is laid out as ``[gw_type2_params..., sr_pmin_0, sr_pmax_0, ...]`` and
      ``cosvf_check_bounds(rtype1_start_idx)`` relies on the index where SR
      parameters begin.  If GW and SR entries are interleaved the index is wrong
      and the EA malfunctions.

   .. note::

      Only ``GW_HF`` and ``GW_KRN`` are classified as **type 0** (no COSVF penalty) because
      pumping links are also constrained to zero (UBC = 0), and so they are inactive.
      All other GW basins — including the Southern California basins (``GW_AV``, ``GW_CH``,
      ``GW_EW``, ``GW_IM``, ``GW_MJ``, ``GW_MWD``, ``GW_OW``, ``GW_SBV``, ``GW_SC``, ``GW_SD``, ``GW_VC``)
      and the Central Valley basins (``GW_01``–``GW_21``) — are type 2.

``inflows.csv``
   Monthly external inflows for the full period of analysis.

   Schema: ``date,j,flow_taf``

``variable-constraints.csv``
   Links with upper/lower bounds that change across water years (e.g. seasonal
   environmental flow requirements).  Identified directly from timeseries bound
   types (``LBT``, ``UBT``, ``EQT``) in the network data.

   Schema: ``date,i,j,k,lower_bound,upper_bound``

   .. important::

      Rows are emitted for **all piecewise segments** (``k = 0 … N-1``), not just
      ``k=0``.  Each segment's bounds are proportional to its share of the total
      physical capacity (resolved via ``_resolve_costs`` / ``_reconcile_step_cost``).
      Sinks and storage self-links are the exception — they only have ``k=0``.


Web App
--------

CALVIN includes a FastAPI + React web app for interactively exploring the network and
optimization results. See the :doc:`app` page for full documentation.

A hosted version is available at `calvin-network-app.onrender.com <https://calvin-network-app.onrender.com>`_.

To run locally:

.. code-block:: bash

   pip install "calvin[app]"
   python -m calvin.app serve --data ../calvin-network-data/data --local
