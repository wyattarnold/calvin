#!/usr/bin/env bash
# =============================================================================
# 01_create_instance.sh — Create the GCP VM for CALVIN COSVF-EA
#
# Usage:
#   export GCP_PROJECT=your-project-id     # required
#   export GCP_ZONE=us-central1-a          # optional, default shown
#   bash scripts/gcp/01_create_instance.sh
# =============================================================================
set -euo pipefail

: "${GCP_PROJECT:?Error: GCP_PROJECT is not set. Run: export GCP_PROJECT=your-project-id}"

INSTANCE="calvin-ea"
ZONE="${GCP_ZONE:-us-central1-a}"
MACHINE_TYPE="n2-highcpu-96"  # 96 vCPUs @ 2.8 GHz. Use c2-standard-60 if C2_CPUS quota is raised to 64+.
# PROVISIONING: STANDARD (~$3.84/hr, no preemption) or SPOT (~$1.05/hr, preemptible)
PROVISIONING="${PROVISIONING:-STANDARD}"
DISK_SIZE="50GB"
DISK_TYPE="pd-balanced"
IMAGE_FAMILY="ubuntu-2204-lts"
IMAGE_PROJECT="ubuntu-os-cloud"

echo "Creating instance: $INSTANCE"
echo "  Project      : $GCP_PROJECT"
echo "  Zone         : $ZONE"
echo "  Machine      : $MACHINE_TYPE ($PROVISIONING)"
echo ""

SCHED_ARGS="--provisioning-model=$PROVISIONING"
if [[ "$PROVISIONING" == "SPOT" ]]; then
  SCHED_ARGS="$SCHED_ARGS --instance-termination-action=STOP"
fi

gcloud compute instances create "$INSTANCE" \
  --project="$GCP_PROJECT" \
  --zone="$ZONE" \
  --machine-type="$MACHINE_TYPE" \
  $SCHED_ARGS \
  --boot-disk-size="$DISK_SIZE" \
  --boot-disk-type="$DISK_TYPE" \
  --image-family="$IMAGE_FAMILY" \
  --image-project="$IMAGE_PROJECT" \
  --scopes=cloud-platform \
  --metadata=enable-oslogin=true

echo ""
echo "Instance created. Wait ~30 seconds for SSH to become available, then run:"
echo "  bash scripts/gcp/02_setup_vm.sh"
