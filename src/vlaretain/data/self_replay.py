"""M7: build a replay dataset from S1's own successful rollouts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from vlaretain.data.splits import STAGE1_SUITES, load_task_table


def successful_frames(rollout_data: dict, episode: int) -> int | None:
    """Number of frames in a rollout's episode up to its first success, or None if it never succeeded."""
    success = rollout_data["success"][episode]
    if not success.any():
        return None
    return int(torch.argmax(success.to(int)).item()) + 1


def to_uint8_hwc(image: torch.Tensor) -> np.ndarray:
    return (image.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()


# UNVERIFIED: needs GPU/LIBERO
def collect_self_replay(
    s1_ckpt: str | Path, out_root: str | Path, *, episodes_per_task: int = 10, n_envs: int = 8, device: str = "cuda",
    n_action_steps: int = 10, eval_seed: int = 777,
) -> int:  # fmt: skip
    """Roll out S1 on every Stage-1 task, and save only successful episodes as a LeRobot dataset."""
    from lerobot.datasets import LeRobotDataset
    from lerobot.policies import make_pre_post_processors
    from lerobot.scripts.lerobot_eval import rollout

    from vlaretain.eval.rollout import EpisodeSpec, chunk_episodes, libero_env_processors, make_vector_env
    from vlaretain.models.load import load_policy

    policy = load_policy(str(s1_ckpt), device=device).eval()
    policy.config.n_action_steps = n_action_steps
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=str(s1_ckpt), preprocessor_overrides={"device_processor": {"device": device}}
    )
    env_pre, env_post = libero_env_processors()
    features = {
        "observation.images.image": {"dtype": "video", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
        "observation.images.image2": {"dtype": "video", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
        "observation.state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
        "action": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
    }
    dataset = LeRobotDataset.create(repo_id="local/self_replay", fps=10, features=features, root=out_root)
    table = load_task_table()
    kept = 0
    for suite in STAGE1_SUITES:
        for task in table[suite]:
            specs = [EpisodeSpec(suite, task["task_id"], i, eval_seed) for i in range(episodes_per_task)]
            for chunk in chunk_episodes(specs, n_envs):
                env = make_vector_env(chunk)
                try:
                    with torch.no_grad():
                        data = rollout(
                            env, policy, env_pre, env_post, preprocessor, postprocessor, return_observations=True,
                            seeds=[eval_seed + s.init_index for s in chunk],
                        )  # fmt: skip
                finally:
                    env.close()
                observations = env_pre(data["observation"])
                for episode in range(len(chunk)):
                    n_frames = successful_frames(data, episode)
                    if n_frames is None:
                        continue
                    for t in range(n_frames):
                        dataset.add_frame({
                            "observation.images.image": to_uint8_hwc(observations["observation.images.image"][episode, t]),
                            "observation.images.image2": to_uint8_hwc(observations["observation.images.image2"][episode, t]),
                            "observation.state": observations["observation.state"][episode, t].cpu().numpy().astype(np.float32),
                            "action": data["action"][episode, t].cpu().numpy().astype(np.float32),
                            "task": task["language"],
                        })  # fmt: skip
                    dataset.save_episode()
                    kept += 1
    dataset.finalize()
    return kept
