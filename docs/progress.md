# Progress

What has been tested, what has not, and the known risks. No training, evaluation, or simulator rollout has been run, and the repository holds no experimental results.

## Status by component

| Component | Status |
| --- | --- |
| Config system and run ids | Tested (CPU) |
| Splits and replay sampling | Tested (CPU) |
| Module groups | Tested (real model, CPU) |
| M0 expert-only SFT | Tested (real model, CPU) |
| M1 full SFT | Tested (real model, CPU) |
| M2 LoRA | Tested (real model, CPU) |
| M3 experience replay | Tested (CPU) |
| M4 weight merging | Tested (CPU) |
| M5 L2-SP | Tested (CPU) |
| M6 feature anchoring | Tested (CPU) |
| M7 self-replay training path | Tested (CPU) |
| M7 rollout collection | Written, needs GPU/LIBERO |
| M8 layer merging | Tested (CPU) |
| Training loop, resume, preemption | Tested (CPU) |
| Eval runner: sharding, skipping, JSONL, in-memory merges | Tested (CPU) |
| Rollouts | Written, needs GPU/LIBERO |
| Perturbations: sampling and model edits | Tested (CPU) |
| Perturbations in the simulator | Written, needs GPU/LIBERO |
| Language tests: swap plan | Tested (CPU) |
| Language tests in the simulator | Written, needs GPU/LIBERO |
| Probes: selection, freezing, round trip | Tested (CPU) |
| Probe building from `lerobot/libero` | Written, needs GPU/LIBERO |
| Weight drift, CKA | Tested (CPU) |
| Divergence metrics | Tested (real model, CPU) |
| Pareto and correlation analysis | Tested (CPU) |
| Figure generation | Tested (CPU) |
| SLURM launch layer | Tested (CPU) |
| Gate checks | Tested (CPU) |

`Tested (CPU)` runs LeRobot's real `SmolVLAPolicy` code at a tiny size (`tests/tiny.py`), or real `LeRobotDataset` files written to a temporary directory (`tests/test_data_pipeline.py`), wherever a model or dataset is involved. `Tested (real model, CPU)` uses `lerobot/smolvla_base` itself, and those tests skip if it is not cached. The SLURM layer was tested with dry-runs and a fake `sbatch`, not on a real scheduler.

## Trainable parameters, measured on `lerobot/smolvla_base`

| Method | Trainable | Vision | Connector | LM early | LM mid | LM late | Expert |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M0 | 99,880,992 | 0 | 0 | 0 | 0 | 0 | 99,880,992 |
| M1 | 306,471,072 | 0 | 11,796,480 | 96,470,400 | 58,993,920 | 39,329,280 | 99,880,992 |
| M2 (r=16) | 7,757,824 | 0 | 0 | 1,356,800 | 1,628,160 | 1,356,800 | 3,416,064 |

The model has 450,046,176 parameters in total.

## UNVERIFIED functions

Each carries the marker `# UNVERIFIED: needs GPU/LIBERO`. None has run against the real simulator or a real GPU. `tests/test_readme.py` checks that this list and the code agree.

- `src/vlaretain/train/loop.py`: `train` (tested on CPU with the tiny model), `make_processors`, `build_train_data`
- `src/vlaretain/eval/rollout.py`: `_retain_env_class`, `build_env`, `make_vector_env`, `libero_env_processors`, `run_episodes`
- `src/vlaretain/eval/runner.py`: `make_libero_task_runner`
- `src/vlaretain/eval/perturb.py`: `apply_sim_perturbation`
- `src/vlaretain/data/self_replay.py`: `collect_self_replay`
- `src/vlaretain/data/probes.py`: `build_probe_file`
- `scripts/drift.py`: `measure`
- `scripts/smoke.py`: `download`, `prepare_libero`, `timing_table`
- `scripts/train.py`: `run_training`

## Known risks

1. **The simulator path has never run.** The smoke test is the first real check of `RetainLiberoEnv`, EGL, and `AsyncVectorEnv` together. Expect fixes there.
2. **robosuite model attributes.** `apply_sim_perturbation` assumes `sim.model.cam_pos`, `light_diffuse`, `mat_texid`, and friends pass through to MuJoCo. Inspect a rendered frame per axis from the eval videos before trusting robustness numbers.
3. **Perturbation strength is unvalidated.** The magnitudes in `configs/eval/robustness.yaml` are our first guesses; tune them on S1 before tuning any method, so they are neither invisible nor destructive.
4. **`n_action_steps=10` is our choice.** The public `smolvla_libero` config uses 1. If S1 falls short at Gate 1, try a smaller value first. It changes eval cost directly.
5. **Dataset fps metadata.** The dataset declares `fps=10`, while the env runs at 20 Hz. Chunk spacing follows LeRobot's defaults, as the public checkpoint does, but it is worth confirming on S1.
6. **Mixed-precision speed.** We train in float32 to keep small updates. If throughput is poor, `train.use_amp=true` uses autocast with bf16 for the forward pass while keeping float32 weights.
7. **Stage-1 trains only the expert.** This is the configuration of the public checkpoint, and it may matter to how much the LM can forget. See `docs/decisions.md`.
8. **Gate 2 sample size.** Tuning yields about 30 checkpoint points from about 15 runs, so its CI will be wide. It is a sanity check, and the paper's claim rests on the finals.
9. **Name overlap.** arXiv:2610.27698 uses the name VLA-Retain for a benchmark on vision tuning. It should be cited as related work, and the owners should decide whether to rename this repository.

## Installing behind a broken git-lfs

On machines where git config references `git-lfs` but the binary is missing, `pip` or `uv` fail while checking out LeRobot. For that one command:

```bash
export GIT_LFS_SKIP_SMUDGE=1 GIT_CONFIG_COUNT=3 GIT_CONFIG_KEY_0=filter.lfs.process GIT_CONFIG_VALUE_0= GIT_CONFIG_KEY_1=filter.lfs.smudge GIT_CONFIG_VALUE_1=cat GIT_CONFIG_KEY_2=filter.lfs.required GIT_CONFIG_VALUE_2=false
```
