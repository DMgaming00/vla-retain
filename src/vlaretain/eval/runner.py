"""Evaluates a checkpoint or an in-memory merge. Shards by task, appends JSONL, and skips finished work."""

from __future__ import annotations

import json
import logging
import socket
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from vlaretain.eval.suites import FULL_EVAL, TaskJob, expand_eval_set
from vlaretain.methods.merge import NO_MERGE, AlphaSpec, merge_state_dicts
from vlaretain.utils.io import (
    append_jsonl,
    completed_eval_keys,
    git_sha,
    lerobot_sha,
    make_eval_row,
)

log = logging.getLogger(__name__)

TaskRunner = Callable[[TaskJob], np.ndarray]


@dataclass
class EvalSource:
    """What is being evaluated: a checkpoint, or an alpha merge of two checkpoints held in memory."""

    run_id: str
    method: str
    seed: int
    step: int
    ckpt: str | Path
    alpha_spec: str = NO_MERGE
    s1_ckpt: str | Path | None = None  # the merge's other endpoint, and the base of adapter checkpoints
    gpu_hours_train: float | None = None


def read_ckpt_meta(ckpt: str | Path) -> dict:
    path = Path(ckpt) / "vlaretain_ckpt.json"
    return json.loads(path.read_text()) if path.exists() else {}


def source_from_run(
    run_root: str | Path, run_id: str, step: int, alpha_spec: str = NO_MERGE, method_label: str | None = None
) -> EvalSource:
    """Describe a checkpoint from what its run recorded on disk."""
    run_dir = Path(run_root) / run_id
    meta = json.loads((run_dir / "run_meta.json").read_text())
    config = meta["config"]
    ckpt = run_dir / "checkpoints" / f"step_{step:06d}"
    s1_run = config.get("s1_run")
    s1_ckpt = Path(run_root) / s1_run / "checkpoints" / f"step_{config['s1_step']:06d}" if s1_run else None
    return EvalSource(
        run_id=run_id,
        method=method_label or config["method"]["name"],
        seed=int(config["seed"]),
        step=step,
        ckpt=ckpt,
        alpha_spec=alpha_spec,
        s1_ckpt=s1_ckpt,
        gpu_hours_train=read_ckpt_meta(ckpt).get("gpu_hours_train"),
    )


def select_jobs(jobs: list[TaskJob], shard: int, num_shards: int) -> list[TaskJob]:
    if not 0 <= shard < num_shards:
        raise ValueError(f"Shard {shard} is outside [0, {num_shards})")
    return jobs[shard::num_shards]


def pending_jobs(jobs: list[TaskJob], source: EvalSource, results_dir: str | Path) -> list[TaskJob]:
    done = completed_eval_keys(results_dir)
    return [j for j in jobs if (source.run_id, source.step, source.alpha_spec, j.eval_set, j.task_key) not in done]


def build_state(source: EvalSource, num_layers: int = 16) -> dict[str, torch.Tensor]:
    """The state dict to evaluate. Merges exist only here, in memory; no code path writes them to disk."""
    from vlaretain.models.load import load_state_for_eval

    state = load_state_for_eval(source.ckpt, source.s1_ckpt)
    if source.alpha_spec == NO_MERGE:
        return state
    if source.s1_ckpt is None:
        raise ValueError("A merge needs s1_ckpt")
    reference = load_state_for_eval(source.s1_ckpt)
    return merge_state_dicts(reference, state, AlphaSpec.parse(source.alpha_spec), num_layers)


def run_jobs(
    source: EvalSource,
    jobs: list[TaskJob],
    results_dir: str | Path,
    make_task_runner: Callable[[], TaskRunner],
    results_name: str | None = None,
) -> int:
    """Run pending jobs and append one row each. The model is loaded only if something is left to do."""
    todo = pending_jobs(jobs, source, results_dir)
    log.info("%d of %d jobs pending for %s", len(todo), len(jobs), source.run_id)
    if not todo:
        return 0
    run_task = make_task_runner()
    name = results_name or f"eval_{source.run_id}_{socket.gethostname()}_{int(time.time())}.jsonl"
    path = Path(results_dir) / name
    code_sha, lib_sha = git_sha(), lerobot_sha()
    for job in todo:
        start = time.monotonic()
        successes = np.asarray(run_task(job), dtype=bool)
        if len(successes) != job.n_episodes:
            raise RuntimeError(f"{job.task_key}: expected {job.n_episodes} results, got {len(successes)}")
        row = make_eval_row(
            run_id=source.run_id, method=source.method, seed=source.seed, step=source.step,
            alpha_spec=source.alpha_spec, eval_set=job.eval_set, task_id=job.task_key,
            n_episodes=job.n_episodes, n_success=int(successes.sum()),
            gpu_hours_train=source.gpu_hours_train, gpu_hours_eval=(time.monotonic() - start) / 3600.0,
            git_sha=code_sha, lerobot_sha=lib_sha,
        )  # fmt: skip
        append_jsonl(path, [row])  # after every task, so a preempted job loses at most one task
    return len(todo)


# UNVERIFIED: needs GPU/LIBERO
def make_libero_task_runner(
    source: EvalSource, *, device: str, n_envs: int, n_action_steps: int, hard_reset: bool, use_amp: bool,
    video_dir: str | Path | None,
) -> TaskRunner:  # fmt: skip
    """Load the policy once, then return a function that evaluates a task with it."""
    from lerobot.policies import make_pre_post_processors

    from vlaretain.eval.rollout import run_episodes
    from vlaretain.models.load import load_policy

    config_dir = str(source.s1_ckpt or source.ckpt)
    policy = load_policy(config_dir, device=device)
    state = build_state(source)
    result = policy.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(f"Unexpected keys: {result.unexpected_keys[:5]}")
    policy.to(device).eval()
    processors = make_pre_post_processors(
        policy.config, pretrained_path=config_dir, preprocessor_overrides={"device_processor": {"device": device}}
    )

    def run(job: TaskJob) -> np.ndarray:
        video = None
        if video_dir is not None:
            video = (
                Path(video_dir) / source.run_id / f"step_{source.step:06d}" / source.alpha_spec.replace(",", "_")
                / job.eval_set / f"{job.suite}_{job.task_id:02d}.mp4"
            )  # fmt: skip
        return run_episodes(
            policy, processors, list(job.specs), n_envs=n_envs, n_action_steps=n_action_steps,
            hard_reset=hard_reset, use_amp=use_amp, video_path=video,
        )  # fmt: skip

    return run


def expand_all(names: tuple[str, ...] | list[str] = FULL_EVAL, config_dir: str | Path = "configs/eval") -> list[TaskJob]:
    return [job for name in names for job in expand_eval_set(name, config_dir)]
