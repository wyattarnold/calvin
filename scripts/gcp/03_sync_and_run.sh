#!/usr/bin/env bash
# =============================================================================
# 03_sync_and_run.sh — Sync code + data to the VM and launch the EA.
#
# Must be run from the calvin repo root.
#
# Usage:
#   export GCP_PROJECT=your-project-id
#   export GCP_ZONE=us-west1-b             # optional, defaults to us-west1-b
#   bash scripts/gcp/03_sync_and_run.sh
#
# To resume from a checkpoint:
#   CHECKPOINT=cosvf-ea-chkpnt-<seed>.pickle bash scripts/gcp/03_sync_and_run.sh
# =============================================================================
set -euo pipefail

: "${GCP_PROJECT:?Error: GCP_PROJECT is not set.}"

INSTANCE="calvin-ea"
ZONE="${GCP_ZONE:-us-west1-b}"
REMOTE="$INSTANCE"
MODEL_DIR="my-models/calvin-cosvf-ea"

# Optional: set CHECKPOINT=<filename> to resume from a specific pickle.
# Leave unset (default) for a cold start.
CHECKPOINT="${CHECKPOINT:-}"
if [[ -n "$CHECKPOINT" ]]; then
  echo "=== Resuming from checkpoint: $CHECKPOINT ==="
else
  echo "=== Cold start ==="
fi

echo "=== Syncing code to $INSTANCE ==="

# Ensure remote directories exist
gcloud compute ssh "$REMOTE" --project="$GCP_PROJECT" --zone="$ZONE" \
  -- "mkdir -p ~/calvin/my-models/calvin-cosvf-ea"

# rsync transport wrapper: exports project/zone so _gcloud_rsync_rsh.sh can read them
export GCP_PROJECT GCP_ZONE="$ZONE"
GCE_RSH="bash $(dirname "$0")/_gcloud_rsync_rsh.sh"

# rsync: skip unchanged files (compares mtime+size), exclude compiled artifacts
rsync -az --update \
  --exclude='__pycache__/' --exclude='*.pyc' --exclude='*.egg-info/' \
  --exclude='app/' \
  -e "$GCE_RSH" calvin/ "${INSTANCE}:~/calvin/calvin/"

rsync -az --update \
  --exclude='__pycache__/' --exclude='*.pyc' \
  -e "$GCE_RSH" scripts/ "${INSTANCE}:~/calvin/scripts/"

rsync -az --update -e "$GCE_RSH" \
  pyproject.toml "${INSTANCE}:~/calvin/pyproject.toml"

echo ""
echo "=== Syncing model input files ==="

rsync -az --update -e "$GCE_RSH" \
  "${MODEL_DIR}/links.csv" \
  "${MODEL_DIR}/r-dict.json" \
  "${MODEL_DIR}/inflows.csv" \
  "${MODEL_DIR}/variable-constraints.csv" \
  "${MODEL_DIR}/cosvf-params.csv" \
  "${INSTANCE}:~/calvin/${MODEL_DIR}/"

# Optionally sync checkpoint for resume
if [[ -n "$CHECKPOINT" && -f "$MODEL_DIR/$CHECKPOINT" ]]; then
  echo "Syncing checkpoint: $CHECKPOINT"
  rsync -az -e "$GCE_RSH" "$MODEL_DIR/$CHECKPOINT" "${INSTANCE}:~/calvin/${MODEL_DIR}/$CHECKPOINT"
fi

echo ""
echo "=== Installing package (editable) ==="
gcloud compute ssh "$REMOTE" --project="$GCP_PROJECT" --zone="$ZONE" \
  -- '~/miniconda/envs/calvin/bin/pip install -q -e ~/calvin/'

echo ""
echo "=== Syncing GCP config.toml ==="
rsync -az -e "$GCE_RSH" \
  scripts/configs/calvin-cosvf-ea-gcp.toml \
  "${INSTANCE}:~/calvin/${MODEL_DIR}/config.toml"

# Patch checkpoint field for resume runs
if [[ -n "$CHECKPOINT" ]]; then
  gcloud compute ssh "$REMOTE" --project="$GCP_PROJECT" --zone="$ZONE" \
    -- "sed -i 's|^checkpoint.*|checkpoint = \"$CHECKPOINT\"|' ~/calvin/${MODEL_DIR}/config.toml"
  echo "Checkpoint patched: $CHECKPOINT"
fi

echo ""
echo "=== Launching EA in tmux session 'ea' ==="
gcloud compute ssh "$REMOTE" --project="$GCP_PROJECT" --zone="$ZONE" -- bash -s <<'ENDSSH'
# Kill any stale session
tmux kill-session -t ea 2>/dev/null || true

# Launch: activate conda env, cd to project, run EA, keep window open on exit
tmux new-session -d -s ea \
  "source ~/miniconda/bin/activate calvin && \
   cd ~/calvin && \
   python scripts/calvin-cosvf-ea.py 2>&1 | tee /tmp/ea-run.log; \
   echo '--- EA finished ---'; bash"

echo "EA started in tmux session 'ea'."
echo "Attach: tmux attach -t ea   (detach: Ctrl-B D)"
ENDSSH

echo ""
echo "Done. Monitor with:"
echo "  gcloud compute ssh $INSTANCE --zone=$ZONE --project=$GCP_PROJECT -- tmux attach -t ea"
echo "  gcloud compute ssh $INSTANCE --zone=$ZONE --project=$GCP_PROJECT -- 'tail -f ~/calvin/my-models/calvin-cosvf-ea/*.log'"
