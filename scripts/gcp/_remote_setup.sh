#!/usr/bin/env bash
# =============================================================================
# _remote_setup.sh — Runs ON the GCP VM to install Miniconda and dependencies.
# Uploaded and executed by 02_setup_vm.sh — do not run locally.
# =============================================================================
set -euo pipefail

CONDA_DIR="$HOME/miniconda"
ENV_NAME="calvin"

echo "=== Installing system packages ==="
sudo apt-get update -q
sudo apt-get install -y -q git tmux wget curl

echo ""
echo "=== Installing Miniconda ==="
if [[ -d "$CONDA_DIR" ]]; then
  echo "Miniconda already installed at $CONDA_DIR — skipping."
else
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$CONDA_DIR"
  rm /tmp/miniconda.sh
fi

# Make conda available in this session
export PATH="$CONDA_DIR/bin:$PATH"

# Add to .bashrc for future sessions
if ! grep -q 'miniconda/bin' ~/.bashrc; then
  echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
fi

echo ""
echo "=== Accepting Anaconda Terms of Service ==="
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

echo ""
echo "=== Creating conda environment: $ENV_NAME (Python 3.12) ==="
if conda env list | grep -q "^$ENV_NAME "; then
  echo "Environment '$ENV_NAME' already exists — skipping creation."
else
  conda create -y -n "$ENV_NAME" python=3.12
fi

echo ""
echo "=== Installing Python packages ==="
# Install the EA dependency group: pyomo, highspy, deap, mpi4py, tqdm, sympy
# plus pandas/numpy/openpyxl for the base package.
# We install here without -e so the package is available even before the
# source is synced; 03_sync_and_run.sh does `pip install -e .` to wire in
# the actual source.
conda run -n "$ENV_NAME" pip install --quiet \
  "pyomo" \
  "highspy" \
  "pandas" \
  "numpy" \
  "openpyxl" \
  "deap" \
  "tqdm" \
  "sympy"

echo ""
echo "=== Verifying HiGHS solver ==="
conda run -n "$ENV_NAME" python -c "
import highspy
from pyomo.contrib import appsi
opt = appsi.solvers.Highs()
print('HiGHS available via APPSI:', opt.available())
"

echo ""
echo "=== Setup complete ==="
