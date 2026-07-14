"""
Limited foresight evolutionary search — optimizes COSVF penalty parameters via NSGA-III.

Model dir: my-models/calvin-cosvf-ea/
  (same input files as calvin-cosvf.py; EA writes checkpoint files here)

To prepare input files:
  python -m calvin.network.cli prepare-cosvf \
      --data ../calvin-network-data/data \
      --output my-models/calvin-cosvf-ea

Run modes
---------
Local (multiprocessing, no MPI needed):
  python scripts/calvin-cosvf-ea.py

MPI cluster (one rank per individual evaluation):
  mpirun -n <ncpus> python scripts/calvin-cosvf-ea.py

The script auto-detects which executor to use:
  - MPI job (COMM_WORLD size > 1) -> MPIPoolExecutor
  - Otherwise                      -> ProcessPoolExecutor with max_workers

Configuration is read from my-models/calvin-cosvf-ea/config.toml (auto-created
from scripts/configs/calvin-cosvf-ea.toml on first run).  Edit the model-dir
copy to customise a specific experiment without touching the committed defaults.
"""
import json, os, sys
from calvin import cosvfea

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_config import load_config

MODEL_DIR = './my-models/calvin-cosvf-ea'
cfg       = load_config('calvin-cosvf-ea', MODEL_DIR)

SOLVER      = cfg['run']['solver']
NPROC       = cfg['run']['nproc']
MAX_WORKERS = cfg['run']['max_workers']
NGEN        = cfg['ea']['n_gen']
MU          = cfg['ea']['mu']
CHECKPOINT  = cfg['ea']['checkpoint'] or None  # empty string -> None

with open(os.path.join(MODEL_DIR, 'r-dict.json')) as _f:
    _r = json.load(_f)
NRTYPE1 = sum(1 for v in _r.values() if v.get('type') == 1)
NRTYPE2 = sum(1 for v in _r.values() if v.get('type') == 2)

# ---------------------------------------------------------------------------
# Worker process state — populated once per process by _worker_init.
# Each worker builds the COSVF model + APPSI solver exactly once on startup,
# so per-evaluation work is only: reset initial storage + assign penalties + solve.
# ---------------------------------------------------------------------------
_worker_calvin = None
_worker_opt    = None
_worker_appsi  = None
_worker_solver = None


def _worker_init(model_dir, solver, nproc):
    """Build and cache COSVF model + solver once per worker process.

    Called by ProcessPoolExecutor via the *initializer* kwarg.  Runs once
    when the worker process starts, avoiding per-evaluation model rebuilds
    (CSV reads, Pyomo construction, APPSI/HiGHS initialisation).

    Each subsequent cosvf_evaluate() call in the same worker reuses the
    cached model, resetting only initial storage and penalty params — the
    dominant cost in the naive approach is the rebuild (~7–8 min/gen vs
    the ~4 s of actual LP solves).
    """
    import logging as _log
    global _worker_calvin, _worker_opt, _worker_appsi, _worker_solver
    _worker_calvin = cosvfea.COSVF(pwd=model_dir, console_level=_log.WARNING)
    _worker_calvin.create_pyomo_model(debug_mode=True)
    _worker_opt, _worker_appsi = cosvfea._init_cosvf_solver(
        solver, nproc, _worker_calvin.log)
    _worker_calvin._capture_initial_storage()
    _worker_solver = solver


def cosvf_evaluate(pcosvf):
    if _worker_calvin is None:
        # Fallback: worker initializer not used (e.g. MPI path or direct call).
        import logging
        calvin = cosvfea.COSVF(pwd=MODEL_DIR, console_level=logging.WARNING)
        calvin.create_pyomo_model(debug_mode=True)
        return calvin.cosvf_solve(solver=SOLVER, nproc=NPROC, pcosvf=pcosvf)
    return _worker_calvin.cosvf_solve_reuse(
        opt=_worker_opt, appsi=_worker_appsi, pcosvf=pcosvf, solver=_worker_solver)

toolbox = cosvfea.cosvf_ea_toolbox(
    cosvf_evaluate=cosvf_evaluate,
    nrtype=[NRTYPE1, NRTYPE2],
    mu=MU,
)

if __name__ == '__main__':
    try:
        from mpi4py import MPI
        _use_mpi = MPI.COMM_WORLD.Get_size() > 1
    except Exception:
        _use_mpi = False

    if _use_mpi:
        from mpi4py.futures import MPIPoolExecutor
        print(f'Using MPIPoolExecutor ({MPI.COMM_WORLD.Get_size()} MPI ranks)')
        with MPIPoolExecutor() as executor:
            toolbox.register('map', executor.map)
            cosvfea.cosvf_ea_main(toolbox=toolbox, n_gen=NGEN, mu=MU, pwd=MODEL_DIR, checkpoint=CHECKPOINT)
    else:
        from concurrent.futures import ProcessPoolExecutor
        print(f'Using ProcessPoolExecutor (max_workers={MAX_WORKERS})')
        with ProcessPoolExecutor(max_workers=MAX_WORKERS,
                                 initializer=_worker_init,
                                 initargs=(MODEL_DIR, SOLVER, NPROC)) as executor:
            toolbox.register('map', executor.map)
            cosvfea.cosvf_ea_main(toolbox=toolbox, n_gen=NGEN, mu=MU, pwd=MODEL_DIR, checkpoint=CHECKPOINT)
