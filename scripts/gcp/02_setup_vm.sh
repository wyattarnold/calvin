#!/usr/bin/env bash
# =============================================================================
# 02_setup_vm.sh — Install Miniconda and all Python dependencies on the VM.
#
# Run once after creating the instance. Takes ~5 minutes.
#
# Usage:
#   export GCP_PROJECT=your-project-id
#   export GCP_ZONE=us-central1-a          # optional
#   bash scripts/gcp/02_setup_vm.sh
# =============================================================================
set -euo pipefail

: "${GCP_PROJECT:?Error: GCP_PROJECT is not set.}"

INSTANCE="calvin-ea"
ZONE="${GCP_ZONE:-us-west1-b}"

echo "Waiting for SSH to become available on $INSTANCE ..."
for i in $(seq 1 20); do
  gcloud compute ssh "$INSTANCE" --project="$GCP_PROJECT" --zone="$ZONE" \
    --ssh-flag="-o ConnectTimeout=5" -- "echo ok" &>/dev/null && break
  echo "  attempt $i/20 — not ready yet, retrying in 10s..."
  sleep 10
done

echo "Setting up VM environment on $INSTANCE ..."

# Upload the remote setup script, then execute it
gcloud compute scp --project="$GCP_PROJECT" --zone="$ZONE" \
  "$(dirname "$0")/_remote_setup.sh" "${INSTANCE}:~/remote_setup.sh"

gcloud compute ssh "$INSTANCE" \
  --project="$GCP_PROJECT" \
  --zone="$ZONE" \
  -- 'bash ~/remote_setup.sh'

echo ""
echo "VM environment ready. Next: bash scripts/gcp/03_sync_and_run.sh"
