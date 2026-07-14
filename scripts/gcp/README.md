# Running CALVIN COSVF-EA on Google Cloud

## Cost & performance summary

| Instance | vCPUs | Spot price | Workers | Time (est.) | Total cost |
|----------|-------|-----------|---------|-------------|------------|
| **n2-highcpu-96** | 96 | ~$1.05/hr | `max_workers` | ~6.5 hrs | **~$7** |
| c2-standard-60 | 60 | ~$0.65/hr | 55 | ~12 hrs | ~$8 |

Default scripts use **n2-highcpu-96 spot** — fastest and cheapest for this workload.
With `max_workers` set equal to `mu`, all individuals in a generation are evaluated in
parallel, so each generation takes approximately one individual eval time (~3 min).

> **Spot preemption:** Only applies if using `PROVISIONING=SPOT`. With `STANDARD` provisioning
> the VM runs uninterrupted. If the VM is ever stopped and restarted, re-run step 3 to
> relaunch the EA — pass `CHECKPOINT=<file>` to resume from a saved generation.

---

## Prerequisites

1. **Install the Google Cloud CLI** (one-time):
   ```bash
   curl https://sdk.cloud.google.com | bash
   exec -l $SHELL
   gcloud init
   ```
   Follow the prompts to log in and select your project.

2. **Set your project ID** in your shell (used by all scripts):
   ```bash
   export GCP_PROJECT=calvin-488815
   export GCP_ZONE=us-west1-b    # optional: change region
   ```

---

## Step-by-step

### 1 — Create the VM

```bash
cd /path/to/calvin
bash scripts/gcp/01_create_instance.sh
```

This creates a spot `n2-highcpu-96` VM named `calvin-ea` in `$GCP_ZONE`.
It will print the external IP when ready.

### 2 — Set up the VM environment

```bash
bash scripts/gcp/02_setup_vm.sh
```

Installs Miniconda and all Python dependencies on the VM (~5 min, run once).

### 3 — Sync code and start the run

```bash
bash scripts/gcp/03_sync_and_run.sh
```

- Rsyncs the `calvin/` package, `scripts/`, and `my-models/calvin-cosvf-ea/` input files.
- Rsyncs `scripts/configs/calvin-cosvf-ea-gcp.toml` as the run `config.toml`. Edit that file to tune `max_workers`, `mu`, `n_gen`, etc.
- Launches the EA in a `tmux` session called `ea` so it survives SSH disconnects.

#### Resuming from a checkpoint

Pass the checkpoint filename as an environment variable:

```bash
CHECKPOINT=cosvf-ea-chkpnt-<seed>.pickle bash scripts/gcp/03_sync_and_run.sh
```

Without `CHECKPOINT` set, the script does a cold start.

### 4 — Monitor progress

```bash
# Attach to the live tmux session
gcloud compute ssh calvin-ea --zone=${GCP_ZONE} --project=$GCP_PROJECT \
  -- tmux attach -t ea

# Or tail the log file directly (Ctrl-C to exit)
gcloud compute ssh calvin-ea --zone=${GCP_ZONE} --project=$GCP_PROJECT \
  -- "tail -f ~/calvin/my-models/calvin-cosvf-ea/*.log"
```

Detach from tmux with `Ctrl-B D` (the run keeps going).

### 5 — Fetch results

```bash
bash scripts/gcp/04_fetch_results.sh
```

Downloads checkpoints and logs from the VM into your local `my-models/calvin-cosvf-ea/`.

### 6 — Delete the VM when finished

```bash
gcloud compute instances delete calvin-ea \
  --zone=${GCP_ZONE} --project=$GCP_PROJECT --quiet
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `gcloud: command not found` | Re-open a new terminal after running `gcloud init` |
| SSH key errors | Run `gcloud compute config-ssh --project=$GCP_PROJECT` |
| VM preempted | `gcloud compute instances start calvin-ea --zone=$GCP_ZONE --project=$GCP_PROJECT` then `bash scripts/gcp/03_sync_and_run.sh` (auto-resumes) |
| `highspy` import error | SSH into VM and run `pip install highspy` in the conda env |
| Out of quota for n2-highcpu-96 | Edit `01_create_instance.sh` and change `MACHINE_TYPE` to `c2-standard-60`; set `max_workers = 55` and `mu = 55` in `scripts/configs/calvin-cosvf-ea-gcp.toml` |
