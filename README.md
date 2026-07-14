# CALVIN

Network flow optimization of California's water supply system.

## Quick Start

Clone the repo and install into your environment. The install step is required so that `calvin` is importable from any working directory (including `scripts/`).

**pip:**
```bash
git clone https://github.com/wyattarnold/calvin.git
cd calvin
pip install -e ".[solver]"
```

**conda:**
```bash
git clone https://github.com/wyattarnold/calvin.git
cd calvin
conda create -n calvin python=3.11
conda activate calvin
pip install -e ".[solver]"
```

> conda users: `pip` works inside conda environments and will install all dependencies. The `-e` flag installs the package in editable mode so that source changes take effect immediately.

Download an archived network (pre-built CSV):
- [1-year example (WY 1922, 400 KB)](https://www.dropbox.com/s/9aq7aaom4dvn0b5/linksWY1922.csv.zip?dl=1)
- [82-year perfect foresight (27 MB)](https://www.dropbox.com/s/ikt5j6kd7n80rir/links82yr.csv.zip?dl=1)
- [Annual, limited foresight — 82 CSV files (31 MB)](https://www.dropbox.com/s/ac1gxs8y49oiw7d/annual.zip?dl=1)

Then run:
```python
from calvin import CALVIN, postprocess

calvin = CALVIN('linksWY1922.csv')
calvin.create_pyomo_model()
calvin.solve_pyomo_model(solver='highs')
postprocess(calvin.df, calvin.model, resultdir='results')
```

Results are written to `results/` as CSV files (flows, storage, shortages, duals, evaporation).

For full model runs, see the ready-to-use scripts in `scripts/` (one per model type).

## Data

The full California network is in [calvin-network-data](https://github.com/ucd-cws/calvin-network-data). Clone it alongside this repo to build custom links files for any time period or spatial subset:

```bash
# Clone alongside this repo (both sit in the same parent directory)
git clone https://github.com/ucd-cws/calvin-network-data

# From within calvin/, build the full 82-year monthly network
python -m calvin.network.cli matrix \
    --data ../calvin-network-data/data \
    --start 1921-10 --stop 2003-09 \
    --output links.csv
```

## App

Explore the California water network and optimization results at [calvin-view.onrender.com](https://calvin-view.onrender.com).

To run locally, `calvin-network-data` is required. Model runs stored in `./my-models/` are loaded automatically as selectable studies; if none exist the app serves the network map only:
```bash
pip install "calvin[app]"
python -m calvin.app serve --data ../calvin-network-data/data --local
```

## Docs

Build the Sphinx docs locally:
```bash
pip install sphinx furo
cd documentation/code && make html
```
Output lands in `documentation/code/build/html/index.html`.
