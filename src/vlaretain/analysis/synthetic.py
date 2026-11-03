"""Fabricated results for the demo figures and the end-to-end test. Not experimental data.

All rates are independent random draws, so no pattern in the demo figures means anything. Run ids start with
`synthetic-`, which lets `scripts/aggregate.py` refuse to place these rows in the project README.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vlaretain.data.splits import STAGE1_SUITES, load_task_table
from vlaretain.eval.perturb import AXES
from vlaretain.models.modules import GROUPS
from vlaretain.utils.io import append_jsonl, make_drift_row, make_eval_row

SYNTHETIC_PREFIX = "synthetic-"
METHODS = ("sft_expert", "sft_full", "lora", "replay", "merge_global", "merge_module", "l2sp")
STEPS = (2000, 5000, 10000, 20000)
ROBUSTNESS = [("libero_spatial", 0), ("libero_object", 3), ("libero_goal", 6)]


def _eval_row(rng, run_id, method, seed, step, alpha, eval_set, task, n, train_hours):
    return make_eval_row(
        run_id=run_id, method=method, seed=seed, step=step, alpha_spec=alpha, eval_set=eval_set, task_id=task,
        n_episodes=n, n_success=int(rng.integers(0, n + 1)), gpu_hours_train=train_hours,
        gpu_hours_eval=float(rng.uniform(0.01, 0.05)), git_sha="synthetic", lerobot_sha="synthetic",
    )  # fmt: skip


def generate_rows(seed: int = 0, seeds: tuple[int, ...] = (1, 2, 3)) -> list[dict]:
    rng = np.random.default_rng(seed)
    table = load_task_table()
    stage1_tasks = [f"{s}/{t['task_id']:02d}" for s in STAGE1_SUITES for t in table[s]]
    target_tasks = [f"libero_10/{t['task_id']:02d}" for t in table["libero_10"]]
    rows = []
    s1_run = f"{SYNTHETIC_PREFIX}s1"
    for task in stage1_tasks:
        rows.append(_eval_row(rng, s1_run, "stage1", 0, 40000, "none", "retention", task, 10, 40.0))
    for method in METHODS:
        for run_seed in seeds:
            run_id = f"{SYNTHETIC_PREFIX}{method}-{run_seed}"
            is_merge = method.startswith("merge")
            alpha = {"merge_global": "global=0.50", "merge_module": "expert=1.00,lm=0.50"}.get(method, "none")
            for step in (20000,) if is_merge else STEPS:
                hours = float(rng.uniform(0.0005, 0.002)) * step
                for task in stage1_tasks:
                    rows.append(_eval_row(rng, run_id, method, run_seed, step, alpha, "retention", task, 10, hours))
                for task in target_tasks:
                    rows.append(_eval_row(rng, run_id, method, run_seed, step, alpha, "target", task, 20, hours))
                for axis in AXES:
                    for suite, task_id in ROBUSTNESS:
                        rows.append(_eval_row(rng, run_id, method, run_seed, step, alpha, f"robustness_{axis}",
                                              f"{suite}/{task_id:02d}", 5, hours))  # fmt: skip
                for task_id in range(10):
                    rows.append(_eval_row(rng, run_id, method, run_seed, step, alpha, "language_swap",
                                          f"libero_goal/{task_id:02d}", 5, hours))  # fmt: skip
                common = dict(
                    run_id=run_id, method=method, seed=run_seed, step=step, alpha_spec=alpha,
                    gpu_hours_train=hours, gpu_hours_eval=0.1, git_sha="synthetic", lerobot_sha="synthetic",
                )  # fmt: skip
                for group in GROUPS:
                    rows.append(make_drift_row(probe_split="all", module=group, layer=None, metric="relative_l2",
                                               value=float(rng.uniform(0.0, 0.2)), **common))  # fmt: skip
                for split in ("new", "old"):
                    for metric in ("action_chunk_distance", "velocity_distance", "action_mmd"):
                        rows.append(make_drift_row(probe_split=split, module="policy", layer=None, metric=metric,
                                                   value=float(rng.uniform(0.01, 1.0)), **common))  # fmt: skip
                    for layer in range(16):
                        rows.append(make_drift_row(probe_split=split, module="lm", layer=layer, metric="cka_distance",
                                                   value=float(rng.uniform(0.0, 0.3)), **common))  # fmt: skip
    return rows


def write_synthetic(out_dir: str | Path, seed: int = 0) -> Path:
    path = Path(out_dir) / "results.jsonl"
    if path.exists():
        path.unlink()
    append_jsonl(path, generate_rows(seed))
    return path
