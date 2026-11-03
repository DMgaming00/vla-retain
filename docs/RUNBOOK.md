# Runbook

Every step from a fresh cluster login to the paper figures. Commands run from the repository root unless noted. All launch scripts accept `--dry-run`, which prints the `sbatch` commands and submits nothing.

## 1. Environment

```bash
git clone <your-repository-url> vla-retain
cd vla-retain
conda create -y -n vla-retain python=3.12
conda activate vla-retain
pip install -e ".[dev,libero]"
```

If `pip` fails with `git-lfs: command not found`, install git-lfs, or export `GIT_LFS_SKIP_SMUDGE=1` and disable the filter for this command as described in `docs/progress.md`.

Check the CPU tests on the login node (LIBERO-specific code is not run by them):

```bash
make test
```

## 2. Fill in `slurm/common.sh`

Replace the five `TODO_` values at the top of the file.

```bash
scontrol show partition          # partition names and limits
sinfo -o "%P %G %l"              # partitions, GPU types, time limits
sacctmgr show assoc user=$USER format=account%30,partition%20,qos%30
echo $SCRATCH                    # candidate for VR_SCRATCH, if your cluster sets it
```

## 3. Check job limits

```bash
scontrol show partition <your-partition> | grep -E "MaxTime|PreemptMode|GraceTime|DefMemPerCPU"
```

If `MaxTime` is below 48 hours, set `VR_TIME` and `VR_EVAL_TIME` in `slurm/common.sh`. If `GraceTime` is under 5 minutes, jobs may be killed before the checkpoint finishes writing; the next link then resumes from the last periodic save.

## 4. Fetch artifacts on the login node

This downloads the models and dataset to scratch, writes LIBERO's config, and fetches its assets. Run it first, because compute nodes may have no internet, and because LIBERO's first import asks a question on stdin.

```bash
source slurm/common.sh
python scripts/smoke.py --download-only
```

## 5. Smoke test

```bash
source slurm/common.sh
mkdir -p "$VR_LOG_DIR"
sbatch --partition="$VR_PARTITION" --account="$VR_ACCOUNT" --gres="gpu:$VR_GPU_TYPE:1" --cpus-per-task="$VR_CPUS" --mem="$VR_MEM" --output="$VR_LOG_DIR/smoke-%j.out" slurm/smoke.sbatch
```

Read its output with `tail -f "$VR_LOG_DIR"/smoke-*.out`. It ends with a table of seconds per episode at 1, 4, and 8 parallel environments, and hours per checkpoint, followed by `SMOKE PASS`. Set `VR_N_ENVS` in `slurm/common.sh` to the fastest row. If the hours per checkpoint would exhaust your allocation, reduce the budget before going on: the table is the basis for that decision. The success-rate column checks only that the pipeline works.

## 6. Stage-1

```bash
bash slurm/launch_stage1.sh --dry-run
bash slurm/launch_stage1.sh
```

This freezes the probe set, trains S1, and queues S1's evaluation automatically.

## 7. Pilots and Gate 1

```bash
bash slurm/launch_pilots.sh --dry-run
bash slurm/launch_pilots.sh
```

When their evals finish:

```bash
python scripts/check_gate1.py
```

The exit code is 0 on PASS, 1 on FAIL (with a suggested fix for each failed condition), and 2 if results are still missing.

## 8. Tuning and Gate 2

```bash
bash slurm/launch_tuning.sh --dry-run
bash slurm/launch_tuning.sh
```

This queues 15 trials with validation evals and drift. A small job then starts the merge sweeps, on the best M1 trial. To run that second part by hand: `bash slurm/launch_tuning.sh --phase merge`.

When everything has finished:

```bash
python scripts/check_gate2.py
```

## 9. Finals

```bash
python scripts/plan_finals.py --dry-run
bash slurm/launch_finals.sh --dry-run
bash slurm/launch_finals.sh
```

The plan is saved to `plans/finals.json`.

## 10. Figures and tables

```bash
python scripts/aggregate.py --results results --out paper_assets
python scripts/aggregate.py --results results --out paper_assets --update-readme
```

Absent results print as `missing`. Partial evals are left out, and never averaged.

## Monitoring

```bash
squeue -u $USER
squeue -u $USER -o "%.12i %.28j %.2t %.10M %.20E"     # includes dependencies
sacct -u $USER -S today --format=JobID,JobName%28,State,Elapsed,ExitCode
tail -f "$VR_LOG_DIR"/vr-s1-*.out
```

Training progress is in `$VR_RUN_ROOT/<run_id>/train_log.jsonl`. A finished run has `$VR_RUN_ROOT/<run_id>/DONE`.

## Resuming after failures

- **Training**: rerun the same launch script. A finished run is skipped, and an unfinished one resumes from `checkpoints/latest`. To extend one chain by hand: `bash slurm/chain.sh --config configs/base.yaml --config configs/methods/sft_full.yaml --trial 1 --set s1_run=<S1_RUN_ID>`.
- **Evals**: rerun with `slurm/launch_evals.sh`. Tasks already in `results/*.jsonl` are skipped: `bash slurm/launch_evals.sh --run-id <RUN_ID> --steps 5000,20000 --sets validation`.
- **A job that fails at once**: read its log, then cancel what depends on it with `scancel`, since `afterany` starts dependents even after a failure. A dependent eval on a missing checkpoint fails quickly and harmlessly.
- **Results**: JSONL files are append-only. Do not edit them. A torn last line from a killed job is ignored by the reader, and that task reruns.

## Cleaning scratch

```bash
du -sh "$VR_BASE"/* | sort -h
# Remove resume state of finished runs. Milestone checkpoints stay.
for run in "$VR_RUN_ROOT"/*; do [ -f "$run/DONE" ] && rm -rf "$run/checkpoints/latest"; done
# Remove videos once you have looked at them.
rm -rf "$VR_VIDEO_DIR"/*
```

Keep S1's checkpoint and `probes/probe_v1.pt` for the life of the project. Every metric refers to both. Check your scratch retention policy, and copy `results/` into git regularly.
