"""Eval sets, loaded from configs/eval/*.yaml and expanded into per-task jobs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from vlaretain.data.splits import STAGE1_SUITES, STAGE2_SUITES, load_task_table
from vlaretain.eval.perturb import AXES, PerturbParams
from vlaretain.eval.rollout import EpisodeSpec

EVAL_SEED = 12345


@dataclass(frozen=True)
class TaskJob:
    """All episodes of one task in one eval set. This is the unit of skipping and sharding."""

    eval_set: str
    suite: str
    task_id: int
    specs: tuple[EpisodeSpec, ...] = field(hash=False)

    @property
    def task_key(self) -> str:
        return task_key(self.suite, self.task_id)

    @property
    def n_episodes(self) -> int:
        return len(self.specs)


def task_key(suite: str, task_id: int) -> str:
    return f"{suite}/{task_id:02d}"


def _specs(suite, task_id, n, offset, seed, axis=None, swap=None, perturb=None) -> tuple[EpisodeSpec, ...]:
    if offset + n > 50:
        raise ValueError(f"LIBERO has 50 init states per task; offset {offset} with {n} episodes runs past them")
    perturb = perturb or PerturbParams()
    return tuple(EpisodeSpec(suite, task_id, offset + i, seed, axis, swap, perturb) for i in range(n))


def _suite_jobs(name, suites, n, offset, seed) -> list[TaskJob]:
    table = load_task_table()
    return [
        TaskJob(name, s, t["task_id"], _specs(s, t["task_id"], n, offset, seed))
        for s in suites
        for t in table[s]
    ]  # fmt: skip


def instruction_swap_map(n_tasks: int = 10) -> dict[int, int]:
    """Give each goal task the next task's instruction. No task keeps its own."""
    return {i: (i + 1) % n_tasks for i in range(n_tasks)}


def load_eval_config(name: str, config_dir: str | Path = "configs/eval") -> dict:
    with open(Path(config_dir) / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def expand_eval_set(name: str, config_dir: str | Path = "configs/eval") -> list[TaskJob]:
    cfg = load_eval_config(name, config_dir)
    seed = int(cfg.get("eval_seed", EVAL_SEED))
    if name == "retention":
        return _suite_jobs("retention", STAGE1_SUITES, cfg["n_episodes"], cfg["init_offset"], seed)
    if name == "target":
        return _suite_jobs("target", STAGE2_SUITES, cfg["n_episodes"], cfg["init_offset"], seed)
    if name == "validation":
        return _suite_jobs(
            "validation_retention", STAGE1_SUITES, cfg["n_episodes"], cfg["init_offset"], seed
        ) + _suite_jobs("validation_target", STAGE2_SUITES, cfg["n_episodes"], cfg["init_offset"], seed)
    if name == "robustness":
        perturb = PerturbParams(**cfg.get("params", {}))
        jobs = []
        for axis in cfg["axes"]:
            if axis not in AXES:
                raise ValueError(f"Unknown axis {axis!r}")
            for item in cfg["tasks"]:
                specs = _specs(
                    item["suite"], item["task_id"], cfg["n_episodes"], cfg["init_offset"], seed, axis=axis, perturb=perturb
                )
                jobs.append(TaskJob(f"robustness_{axis}", item["suite"], item["task_id"], specs))
        return jobs
    if name == "language":
        # libero_goal success is read from the retention rows, so only the swap test costs rollouts here.
        swap = instruction_swap_map()
        return [
            TaskJob(
                "language_swap", "libero_goal", i,
                _specs("libero_goal", i, cfg["n_episodes"], cfg["init_offset"], seed, swap=swap[i]),
            )
            for i in swap
        ]  # fmt: skip
    raise KeyError(f"Unknown eval set {name!r}")


FULL_EVAL = ("retention", "target", "robustness", "language")


def validation_init_indices(config_dir: str | Path = "configs/eval") -> set[int]:
    return {s.init_index for j in expand_eval_set("validation", config_dir) for s in j.specs}


def heldout_init_indices(config_dir: str | Path = "configs/eval") -> set[int]:
    return {s.init_index for n in FULL_EVAL for j in expand_eval_set(n, config_dir) for s in j.specs}
