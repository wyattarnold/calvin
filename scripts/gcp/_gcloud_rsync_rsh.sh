#!/usr/bin/env bash
# _gcloud_rsync_rsh.sh — rsync transport wrapper for gcloud compute ssh
#
# rsync -e calls this as:  <script> <hostname> rsync --server [args...]
# We strip the hostname (first arg) and forward the rest via gcloud compute ssh.
#
# Requires GCP_PROJECT and GCP_ZONE to be set in the environment (03_sync_and_run.sh
# and 04_fetch_results.sh export them before invoking rsync).
HOST="$1"; shift
exec gcloud compute ssh "$HOST" \
  --project="${GCP_PROJECT}" \
  --zone="${GCP_ZONE:-us-central1-a}" \
  -- "$@"
