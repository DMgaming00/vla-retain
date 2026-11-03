"""Rollouts with fixed init states: LeRobot's own `rollout`, driven by explicit per-episode specs."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vlaretain.eval.perturb import PerturbParams, add_image_noise, apply_sim_perturbation, episode_rng
from vlaretain.utils.seed import derive_seed

log = logging.getLogger(__name__)

OBS_SIZE = 256  # matches the dataset's frames


@dataclass(frozen=True)
class EpisodeSpec:
    suite: str
    task_id: int
    init_index: int
    eval_seed: int
    axis: str | None = None
    swap_task_id: int | None = None  # instruction swap: give this task's language, and judge by its goal
    perturb: PerturbParams = field(default_factory=PerturbParams)


def chunk_episodes(specs: list[EpisodeSpec], n_envs: int) -> list[list[EpisodeSpec]]:
    if n_envs < 1:
        raise ValueError("n_envs must be at least 1")
    return [specs[i : i + n_envs] for i in range(0, len(specs), n_envs)]


def episode_successes(rollout_data: dict[str, torch.Tensor]) -> np.ndarray:
    """Per-episode success, following lerobot_eval: mask everything after the first done, then take `any`."""
    done, success = rollout_data["done"], rollout_data["success"]
    n_steps = done.shape[1]
    first_done = torch.argmax(done.to(int), dim=1)
    mask = torch.arange(n_steps)[None, :] <= (first_done[:, None] + 1)
    return (success.bool() & mask).any(dim=1).numpy()


# UNVERIFIED: needs GPU/LIBERO
def _retain_env_class():
    from lerobot.envs.libero import LiberoEnv

    class RetainLiberoEnv(LiberoEnv):
        """LiberoEnv plus render perturbations and instruction swaps. LeRobot itself is unchanged."""

        def configure(self, spec: EpisodeSpec, swap_language: str | None, swap_goal: list | None) -> None:
            self._spec = spec
            self._swap_goal = swap_goal
            if swap_language is not None:
                self.task_description = swap_language
            self._noise_rng = None

        def _format_raw_obs(self, raw_obs):
            obs = super()._format_raw_obs(raw_obs)
            spec = getattr(self, "_spec", None)
            if spec is not None and spec.axis == "noise":
                if self._noise_rng is None:
                    self._noise_rng = episode_rng(spec.eval_seed, spec.suite, spec.task_id, spec.init_index, "noise")
                obs["pixels"] = {
                    k: add_image_noise(v, self._noise_rng, spec.perturb.noise_std) for k, v in obs["pixels"].items()
                }
            return obs

        def reset(self, seed=None, **kwargs):
            observation, info = super().reset(seed=seed, **kwargs)
            spec = getattr(self, "_spec", None)
            if spec is None:
                return observation, info
            inner = self._env.env
            if self._swap_goal is not None:
                inner.parsed_problem["goal_state"] = self._swap_goal
            if spec.axis in ("camera", "lighting", "background"):
                # A hard reset rebuilds the model, so we re-apply the same edit on each reset.
                rng = episode_rng(spec.eval_seed, spec.suite, spec.task_id, spec.init_index, spec.axis)
                apply_sim_perturbation(inner.sim, spec.axis, rng, spec.perturb)
                inner.sim.forward()
                raw_obs = inner._get_observations(force_update=True)
                observation = self._format_raw_obs(raw_obs)
            return observation, info

    return RetainLiberoEnv


# UNVERIFIED: needs GPU/LIBERO
def build_env(spec: EpisodeSpec, hard_reset: bool = True, control_mode: str = "relative"):
    """Build one environment. Runs inside the worker process, which is where EGL has to be initialized."""
    import os

    from lerobot.envs.libero import _get_suite
    from libero.libero import get_libero_path
    from libero.libero.envs import bddl_utils

    suite = _get_suite(spec.suite)
    env = _retain_env_class()(
        task_suite=suite,
        task_id=spec.task_id,
        task_suite_name=spec.suite,
        obs_type="pixels_agent_pos",
        observation_width=OBS_SIZE,
        observation_height=OBS_SIZE,
        init_states=True,
        episode_index=spec.init_index,
        n_envs=1,
        control_mode=control_mode,
        hard_reset=hard_reset,
    )
    swap_language, swap_goal = None, None
    if spec.swap_task_id is not None:
        other = suite.get_task(spec.swap_task_id)
        swap_language = other.language
        bddl = os.path.join(get_libero_path("bddl_files"), other.problem_folder, other.bddl_file)
        swap_goal = bddl_utils.robosuite_parse_problem(bddl)["goal_state"]
    env.configure(spec, swap_language, swap_goal)
    return env


# UNVERIFIED: needs GPU/LIBERO
def make_vector_env(specs: list[EpisodeSpec], hard_reset: bool = True):
    import gymnasium as gym
    from lerobot.envs.utils import freeze_after_episode_end

    fns = [freeze_after_episode_end(partial(build_env, spec, hard_reset)) for spec in specs]
    return gym.vector.AsyncVectorEnv(
        fns, context="forkserver", shared_memory=True, autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP
    )


# UNVERIFIED: needs GPU/LIBERO
def libero_env_processors():
    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig

    return LiberoEnvConfig().get_env_processors()


# UNVERIFIED: needs GPU/LIBERO
def run_episodes(
    policy,
    processors: tuple[Any, Any],
    specs: list[EpisodeSpec],
    *,
    n_envs: int,
    n_action_steps: int,
    hard_reset: bool = True,
    use_amp: bool = False,
    video_path: Path | None = None,
) -> np.ndarray:
    """Run the given episodes in parallel chunks and return one success flag per spec, in order.

    With a video path, episode 0 of the task (the first spec) is written to disk.
    """
    from lerobot.envs.utils import NEW_ROLLOUT_OPTION  # noqa: F401  (rollout passes it to env.reset)
    from lerobot.scripts.lerobot_eval import rollout
    from lerobot.utils.io_utils import write_video

    env_pre, env_post = libero_env_processors()
    preprocessor, postprocessor = processors
    policy.config.n_action_steps = n_action_steps
    policy.eval()
    device_type = next(policy.parameters()).device.type
    results = []
    for index, chunk in enumerate(chunk_episodes(specs, n_envs)):
        first = chunk[0]
        torch.manual_seed(derive_seed("eval", first.eval_seed, first.suite, first.task_id, first.axis, index))
        env = make_vector_env(chunk, hard_reset)
        frames: list[np.ndarray] = []

        def record(vec_env, store=frames):
            store.append(np.asarray(vec_env.call("render")[0]))

        callback = record if (video_path is not None and index == 0) else None
        seeds = [derive_seed("env", s.eval_seed, s.suite, s.task_id, s.init_index) for s in chunk]
        try:
            amp = torch.autocast(device_type, dtype=torch.bfloat16, enabled=use_amp)
            with torch.no_grad(), amp:
                data = rollout(
                    env, policy, env_pre, env_post, preprocessor, postprocessor, seeds=seeds, render_callback=callback
                )
        finally:
            env.close()
        results.append(episode_successes(data))
        if frames:
            video_path.parent.mkdir(parents=True, exist_ok=True)
            write_video(str(video_path), np.stack(frames), fps=20)
    return np.concatenate(results) if results else np.zeros(0, dtype=bool)
