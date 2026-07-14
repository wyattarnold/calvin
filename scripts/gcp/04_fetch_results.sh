#!/usr/bin/env bash
# =============================================================================
# 04_fetch_results.sh — Download checkpoints and logs from the VM.
#
# Safe to run while the EA is still running — does not affect the remote files.
#
# Usage:
#   export GCP_PROJECT=your-project-id
#   export GCP_ZONE=us-central1-a          # optional
#   bash scripts/gcp/04_fetch_results.sh
# =============================================================================
set -euo pipefail

: "${GCP_PROJECT:?Error: GCP_PROJECT is not set.}"

INSTANCE="calvin-ea"
ZONE="${GCP_ZONE:-us-west1-b}"
LOCAL_DIR="my-models/calvin-cosvf-ea"

echo "=== Fetching results from $INSTANCE ==="

export GCP_PROJECT GCP_ZONE="$ZONE"
GCE_RSH="bash $(dirname "$0")/_gcloud_rsync_rsh.sh"

# rsync: pull only checkpoints, logs, and result CSVs; skip input files
rsync -az --update \
  --include='*.pickle' \
  --include='*.log' \
  --include='results/' \
  --include='results/**' \
  --exclude='links.csv' \
  --exclude='inflows.csv' \
  --exclude='variable-constraints.csv' \
  --exclude='cosvf-params.csv' \
  --exclude='r-dict.json' \
  --exclude='links-pyomo-model-reference.csv' \
  --exclude='*' \
  -e "$GCE_RSH" \
  "${INSTANCE}:~/calvin/${LOCAL_DIR}/" \
  "${LOCAL_DIR}/"

echo "Done. Files in $LOCAL_DIR/:"
ls -lh "$LOCAL_DIR/"*.pickle "$LOCAL_DIR/"*.log 2>/dev/null || echo "  (none yet)"
