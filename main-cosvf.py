import os, shutil

from calvin import cosvfea

# CONFIG
COSVF_DIR = './my-models/calvin-lf-test'     # DIRECTORY FOR LINKS, INFLOWS, SR_DICT etc.
SOLVER, NPROC = 'cbc', 1             # SOLVER AND CPU

# construct results path from run name
result_dir = os.path.join(COSVF_DIR,'results')
if not os.path.isdir(result_dir): os.makedirs(result_dir)

# calvin instance
calvin = cosvfea.COSVF(pwd=COSVF_DIR)
calvin.create_pyomo_model(debug_mode=True)

# solve model
f1, f2, f3 = calvin.cosvf_solve(solver=SOLVER, nproc=NPROC, resultdir=result_dir)
print('total cost={} $M/yr'.format(f1))
print('gw overdraft ={} MAF/yr'.format(f2))

# copy the COSVF params used to the results directory
shutil.copy(os.path.join(COSVF_DIR,'cosvf-params.csv'), 
            os.path.join(result_dir,'cosvf-params.csv'))
