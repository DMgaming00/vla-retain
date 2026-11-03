# Compute notes

We plan for the SJSU College of Engineering HPC cluster: SLURM, 4 shared H100/A100 nodes, assumed 48-hour job limits, and condo nodes that preempt with about 5 minutes' warning. We have not yet confirmed these on the cluster. Check them with `scontrol show partition`.

## Why evaluation dominates cost

A training step is a fixed cost. An eval is 750 rollouts per checkpoint, each up to 280-520 environment steps with rendering, physics, and a batched denoising pass every `n_action_steps` steps. Tuning, merge sweeps, and finals all multiply that number. The parallel environment count is therefore the most important tuning knob in the project, and `scripts/smoke.py` measures it before anything else runs. Set `VR_N_ENVS` from its table.

Rollout counts by phase, computed from the configs (the runtime depends on the smoke test):

| Phase | Rollouts |
| --- | --- |
| S1 eval | 750 + 200 validation |
| One tuning trial (validation at 2 steps) | 400 |
| Tuning, 15 trials | 6,000 |
| Merge sweep, global grid (11 alphas) | 2,200 |
| Merge sweep, module grid (16 alphas) | 3,200 |
| One final run (4 checkpoints) | 3,000 |
| Finals, 5 methods x 3 seeds | 45,000 |
| Finals, merges (2 grids x 3 seeds) | 4,500 |

## Preemption and time limits

- `train.sbatch` asks for `--signal=B:USR1@300`. SLURM signals the batch shell 300 s before the time limit, and the shell forwards the signal to Python. The loop finishes its current step, saves `checkpoints/latest`, and exits.
- Preemption may arrive as SIGTERM rather than SIGUSR1. The shell forwards that too, and Python handles both. If the cluster's grace time is shorter than one step plus one save, the next link resumes from the last periodic save instead, losing at most 1k steps.
- `chain.sh` queues links joined by `afterany`. Links exit at once when the `DONE` marker exists.
- Eval and drift jobs use `--requeue`. They append per task or per checkpoint, and skip what is already present, so repeats are harmless.

## Storage

Checkpoints go to `$VR_SCRATCH/vla-retain/runs`, not the repository.

| Item | Size | Count |
| --- | --- | --- |
| Full checkpoint (float32) | about 1.8 GB | 4 per run (LoRA: adapter only, measured count 7.76M parameters, about 31 MB) |
| Rolling `latest` for M1-style runs | about 4.3 GB (weights + AdamW state for 306M parameters) | 1 per run while training |
| Probe set | about 200 MB (uint8 frames) | 1 |
| `lerobot/libero` | 1.9 GB per its documentation | 1 |

The optimizer state of finished runs sits in `checkpoints/latest`. It can be deleted once a run has `DONE`. See the cleaning section of `docs/RUNBOOK.md`.

## Offline compute nodes

If compute nodes have no internet, run `python scripts/smoke.py --download-only` on the login node first, then export `HF_HUB_OFFLINE=1` in `slurm/common.sh`.
