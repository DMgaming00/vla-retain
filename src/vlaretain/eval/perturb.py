"""Our own perturbation wrapper for the four robustness axes.

Camera, lighting, and background edits change only how the scene renders, so they are applied after the settling
steps of a reset without affecting the physics. See docs/decisions.md for why we do not use LIBERO-Plus.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

AXES = ("camera", "lighting", "background", "noise")
_BACKGROUND_WORDS = ("table", "floor", "wall")


@dataclass(frozen=True)
class PerturbParams:
    camera_pos_m: float = 0.03
    camera_rot_deg: float = 5.0
    light_scale_low: float = 0.4
    light_scale_high: float = 1.6
    light_pos_m: float = 0.5
    background_tint_low: float = 0.5
    background_tint_high: float = 1.3
    noise_std: float = 0.03  # as a fraction of the 0-255 pixel range
    agentview_camera: str = "agentview"

    def to_dict(self) -> dict:
        return asdict(self)


def episode_rng(eval_seed: int, suite: str, task_id: int, init_index: int, axis: str) -> np.random.Generator:
    """A perturbation depends only on the episode's identity, so every method sees the same one."""
    axis_code = AXES.index(axis)
    suite_code = sum(ord(c) for c in suite)
    return np.random.default_rng([eval_seed, suite_code, task_id, init_index, axis_code])


def add_image_noise(image: np.ndarray, rng: np.random.Generator, std: float) -> np.ndarray:
    """Gaussian noise on a uint8 image."""
    noisy = image.astype(np.float32) + rng.normal(0.0, std * 255.0, image.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def small_rotation(rng: np.random.Generator, max_deg: float) -> np.ndarray:
    """A unit quaternion (w, x, y, z) for a random axis and an angle within +-max_deg."""
    axis = rng.normal(size=3)
    axis = axis / np.linalg.norm(axis)
    half = np.deg2rad(rng.uniform(-max_deg, max_deg)) / 2.0
    return np.concatenate([[np.cos(half)], np.sin(half) * axis])


def perturb_camera(model, camera_id: int, rng: np.random.Generator, params: PerturbParams) -> None:
    model.cam_pos[camera_id] = model.cam_pos[camera_id] + rng.uniform(-params.camera_pos_m, params.camera_pos_m, 3)
    rotated = _quat_multiply(small_rotation(rng, params.camera_rot_deg), np.asarray(model.cam_quat[camera_id]))
    model.cam_quat[camera_id] = rotated / np.linalg.norm(rotated)


def perturb_lighting(model, rng: np.random.Generator, params: PerturbParams) -> None:
    for i in range(len(model.light_diffuse)):
        scale = rng.uniform(params.light_scale_low, params.light_scale_high)
        model.light_diffuse[i] = np.clip(np.asarray(model.light_diffuse[i]) * scale, 0.0, 1.0)
        model.light_pos[i] = model.light_pos[i] + rng.uniform(-params.light_pos_m, params.light_pos_m, 3)


def background_material_ids(material_names: list[str | None]) -> list[int]:
    return [i for i, n in enumerate(material_names) if n and any(w in n.lower() for w in _BACKGROUND_WORDS)]


def perturb_background(model, material_ids: list[int], rng: np.random.Generator, params: PerturbParams) -> None:
    """Permute textures among background materials and tint them. MuJoCo reads both when it builds each frame."""
    if not material_ids:
        raise RuntimeError("No background materials found; the name filter needs updating")
    originals = {i: np.array(model.mat_texid[i]) for i in material_ids}
    permuted = rng.permutation(material_ids)
    for target, source in zip(material_ids, permuted, strict=True):
        model.mat_texid[target] = originals[int(source)]
        tint = rng.uniform(params.background_tint_low, params.background_tint_high, 3)
        rgba = np.array(model.mat_rgba[target])
        rgba[:3] = np.clip(rgba[:3] * tint, 0.0, 1.0)
        model.mat_rgba[target] = rgba


# UNVERIFIED: needs GPU/LIBERO
def apply_sim_perturbation(sim, axis: str, rng: np.random.Generator, params: PerturbParams) -> None:
    """Edit a robosuite sim's model for one axis. Image noise is applied per frame by the env instead."""
    import mujoco

    model = sim.model
    if axis == "camera":
        perturb_camera(model, model.camera_name2id(params.agentview_camera), rng, params)
    elif axis == "lighting":
        perturb_lighting(model, rng, params)
    elif axis == "background":
        raw = model._model
        names = [mujoco.mj_id2name(raw, mujoco.mjtObj.mjOBJ_MATERIAL, i) for i in range(raw.nmat)]
        perturb_background(model, background_material_ids(names), rng, params)
    elif axis != "noise":
        raise ValueError(f"Unknown perturbation axis {axis!r}")
