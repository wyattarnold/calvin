#!/usr/bin/env bash
# Build script for Render (and general CI) deployment.
# Compiles the React frontend so it can be served by FastAPI.
set -euo pipefail

# Install Python package in editable mode so calvin.app is importable.
# (Running from calvin/app/; repo root is two levels up.)
pip install -e "../../.[app]" --quiet

# Download bundled data.zip from GitHub Releases (too large for git).
# Set DATA_ZIP_URL in Render's environment variables to the release asset URL.
if [ -n "${DATA_ZIP_URL:-}" ]; then
  echo "Downloading data.zip from $DATA_ZIP_URL ..."
  curl -fsSL "$DATA_ZIP_URL" -o data.zip
  echo "data.zip downloaded ($(du -sh data.zip | cut -f1))."
else
  echo "Warning: DATA_ZIP_URL not set — hosted mode will have no bundled data."
fi

# Build the React frontend (outputs to static/).
cd frontend
npm ci --prefer-offline
npm run build
cd -

echo "Build complete."
