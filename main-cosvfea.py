"""
Main Open-MPI parallel processing script for COSVF evolutionary search. 
Script creates a CALVIN instance and DEAP evolutionary algorithm (NSGA-III) toolbox. 
Evolution stops after user-specified number of generations.
"""
from mpi4py.futures import MPIPoolExecutor
from calvin import cosvfea


COSVF_DIR='./my-models/calvin-lf'  # (path) Directory of Calvin model for COSVF search
NRTYPE1=26          # (int) Number of quadratic COSVF reservoirs
NRTYPE2=32          # (int) Number of linear COSVF reservoirs
SOLVER='gurobi'     # (string) Solver for Pyomo model
NPROC=1             # (int) Number of processors to allocate to each solver
SEED=2              # (int) Random seed
NGEN=1              # (int) Number of evolutionary generations to conduct
MU=8                # (int) Population count

# calvin evaluation function for ea
def cosvf_evaluate(pcosvf):
    calvin = cosvfea.COSVF(pwd=COSVF_DIR)
    calvin.create_pyomo_model(debug_mode=True)
    return calvin.cosvf_solve(solver=SOLVER, nproc=NPROC, pcosvf=pcosvf)

# create a DEAP ea toolbox
toolbox = cosvfea.cosvf_ea_toolbox(cosvf_evaluate=cosvf_evaluate, nrtype=[NRTYPE1, NRTYPE2], mu=MU)

# run EA                                        
if __name__ == "__main__":
    # register cpu pool with toolbox
    with MPIPoolExecutor() as executor:
        toolbox.register("map", executor.map)
        # conduct cosvf search (output as a DEAP logbook of evolutionary process)
        cosvfea.cosvf_ea_main(toolbox=toolbox, n_gen=NGEN, mu=MU, seed=SEED, pwd=COSVF_DIR)
