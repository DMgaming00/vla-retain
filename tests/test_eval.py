import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.tiny import NUM_LAYERS, make_tiny_policy
from vlaretain.eval import perturb
from vlaretain.eval.rollout import EpisodeSpec, chunk_episodes, episode_successes
from vlaretain.eval.runner import EvalSource, build_state, expand_all, pending_jobs, run_jobs, select_jobs
from vlaretain.eval.suites import (
    FULL_EVAL,
    expand_eval_set,
    heldout_init_indices,
    instruction_swap_map,
    validation_init_indices,
)
from vlaretain.methods.merge import AlphaSpec, merge_state_dicts
from vlaretain.utils.io import RESULT_FIELDS, read_results

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "eval"


def _jobs(name):
    return expand_eval_set(name, CONFIG_DIR)


def test_episode_budget_matches_the_protocol():
    counts = {n: sum(j.n_episodes for j in _jobs(n)) for n in FULL_EVAL}
    assert counts == {"retention": 300, "target": 200, "robustness": 200, "language": 50}
    assert len(_jobs("retention")) == 30 and len(_jobs("target")) == 10


def test_robustness_has_ten_tasks_on_four_axes():
    jobs = _jobs("robustness")
    assert {j.eval_set for j in jobs} == {f"robustness_{a}" for a in perturb.AXES}
    for axis in perturb.AXES:
        assert len([j for j in jobs if j.eval_set == f"robustness_{axis}"]) == 10
    assert all(s.axis in perturb.AXES for j in jobs for s in j.specs)


def test_validation_init_states_are_disjoint_from_every_test_set():
    assert not validation_init_indices(CONFIG_DIR) & heldout_init_indices(CONFIG_DIR)
    assert validation_init_indices(CONFIG_DIR)


def test_fixed_seed_and_init_states_across_expansions():
    assert _jobs("target")[0].specs == _jobs("target")[0].specs
    specs = _jobs("target")[3].specs
    assert [s.init_index for s in specs] == list(range(20))
    assert {s.eval_seed for j in expand_all(FULL_EVAL, CONFIG_DIR) for s in j.specs} == {12345}


def test_instruction_swap_is_a_derangement():
    swap = instruction_swap_map()
    assert sorted(swap.values()) == list(range(10))
    assert all(k != v for k, v in swap.items())
    assert all(s.swap_task_id == (s.task_id + 1) % 10 for j in _jobs("language") for s in j.specs)


def test_too_many_episodes_are_rejected(tmp_path):
    (tmp_path / "target.yaml").write_text("n_episodes: 20\ninit_offset: 40\n")
    with pytest.raises(ValueError, match="50 init states"):
        expand_eval_set("target", tmp_path)


def test_episode_success_masks_after_first_done():
    done = torch.tensor([[0, 1, 1, 1], [0, 0, 0, 1], [0, 0, 0, 1]], dtype=torch.bool)
    success = torch.tensor([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]], dtype=torch.bool)
    assert episode_successes({"done": done, "success": success}).tolist() == [True, False, True]


def test_chunking_keeps_order_and_rejects_zero():
    specs = [EpisodeSpec("s", 0, i, 0) for i in range(5)]
    assert [len(c) for c in chunk_episodes(specs, 2)] == [2, 2, 1]
    assert [s for c in chunk_episodes(specs, 2) for s in c] == specs
    with pytest.raises(ValueError):
        chunk_episodes(specs, 0)


# ---------- perturbations ----------
def test_episode_rng_is_deterministic_and_axis_specific():
    a = perturb.episode_rng(1, "libero_goal", 2, 3, "camera").random(4)
    assert np.array_equal(a, perturb.episode_rng(1, "libero_goal", 2, 3, "camera").random(4))
    assert not np.array_equal(a, perturb.episode_rng(1, "libero_goal", 2, 3, "lighting").random(4))
    assert not np.array_equal(a, perturb.episode_rng(1, "libero_goal", 2, 4, "camera").random(4))


def test_image_noise_stays_in_range_and_changes_pixels():
    image = np.full((8, 8, 3), 250, dtype=np.uint8)
    noisy = perturb.add_image_noise(image, np.random.default_rng(0), 0.1)
    assert noisy.dtype == np.uint8 and noisy.max() <= 255 and not np.array_equal(noisy, image)


def _fake_model():
    return SimpleNamespace(
        cam_pos=np.zeros((2, 3)),
        cam_quat=np.tile(np.array([1.0, 0, 0, 0]), (2, 1)),
        light_diffuse=np.full((2, 3), 0.5),
        light_pos=np.zeros((2, 3)),
        mat_rgba=np.full((4, 4), 0.5),
        mat_texid=np.arange(4).reshape(4, 1),
    )


def test_camera_perturbation_moves_only_that_camera_and_keeps_a_unit_quaternion():
    model = _fake_model()
    perturb.perturb_camera(model, 1, np.random.default_rng(0), perturb.PerturbParams())
    assert np.all(model.cam_pos[0] == 0) and np.any(model.cam_pos[1] != 0)
    assert np.abs(model.cam_pos[1]).max() <= 0.03
    assert np.linalg.norm(model.cam_quat[1]) == pytest.approx(1.0)
    angle = 2 * np.degrees(np.arccos(np.clip(model.cam_quat[1][0], -1, 1)))
    assert angle <= 5.0 + 1e-6


def test_lighting_and_background_perturbations():
    model = _fake_model()
    perturb.perturb_lighting(model, np.random.default_rng(0), perturb.PerturbParams())
    assert model.light_diffuse.max() <= 1.0 and not np.allclose(model.light_diffuse, 0.5)
    assert perturb.background_material_ids(["Table_mat", None, "robot", "floorplank"]) == [0, 3]
    perturb.perturb_background(model, [0, 3], np.random.default_rng(1), perturb.PerturbParams())
    assert sorted(model.mat_texid[[0, 3]].ravel().tolist()) == [0, 3]
    assert np.array_equal(model.mat_rgba[1], [0.5, 0.5, 0.5, 0.5])
    with pytest.raises(RuntimeError):
        perturb.perturb_background(model, [], np.random.default_rng(1), perturb.PerturbParams())


# ---------- runner ----------
def _source(**kw):
    defaults = dict(run_id="abc", method="sft_full", seed=0, step=2000, ckpt="unused", gpu_hours_train=1.5)
    return EvalSource(**{**defaults, **kw})


def test_shards_partition_the_jobs():
    jobs = _jobs("retention")
    shards = [select_jobs(jobs, i, 4) for i in range(4)]
    keys = [j.task_key for s in shards for j in s]
    assert sorted(keys) == sorted(j.task_key for j in jobs) and len(set(keys)) == len(keys)
    with pytest.raises(ValueError):
        select_jobs(jobs, 4, 4)


def test_rows_have_every_required_field_and_finished_work_is_skipped(tmp_path):
    jobs = _jobs("target")[:3]
    calls = []

    def factory():
        def run(job):
            calls.append(job.task_key)
            return np.array([True] * 5 + [False] * (job.n_episodes - 5))

        return run

    assert run_jobs(_source(), jobs, tmp_path, factory) == 3
    rows = read_results(tmp_path)
    assert len(rows) == 3
    for row in rows:
        assert set(RESULT_FIELDS) <= set(row)
        assert row["n_episodes"] == 20 and row["n_success"] == 5 and row["sr"] == 0.25
        assert row["wilson_lo"] < 0.25 < row["wilson_hi"]
        assert row["gpu_hours_train"] == 1.5 and row["alpha_spec"] == "none" and row["task_id"].startswith("libero_10/")

    calls.clear()
    built = []
    assert run_jobs(_source(), jobs, tmp_path, lambda: built.append(1) or factory()) == 0
    assert calls == [] and built == [], "with nothing to do, the model must not even load"


def test_skip_key_includes_step_alpha_and_eval_set(tmp_path):
    jobs = _jobs("target")[:1]
    factory = lambda: (lambda job: np.zeros(job.n_episodes, dtype=bool))  # noqa: E731
    run_jobs(_source(), jobs, tmp_path, factory)
    assert pending_jobs(jobs, _source(), tmp_path) == []
    assert pending_jobs(jobs, _source(step=5000), tmp_path) == jobs
    assert pending_jobs(jobs, _source(alpha_spec="global=0.50"), tmp_path) == jobs
    assert pending_jobs(jobs, _source(run_id="zzz"), tmp_path) == jobs


def test_interrupted_job_only_reruns_the_missing_tasks(tmp_path):
    jobs = _jobs("target")[:4]
    seen = []

    def crashing_factory():
        def run(job):
            if len(seen) == 2:
                raise RuntimeError("preempted")
            seen.append(job.task_key)
            return np.zeros(job.n_episodes, dtype=bool)

        return run

    with pytest.raises(RuntimeError):
        run_jobs(_source(), jobs, tmp_path, crashing_factory)
    assert len(read_results(tmp_path)) == 2
    seen.clear()
    assert run_jobs(_source(), jobs, tmp_path, lambda: (lambda job: np.zeros(job.n_episodes, dtype=bool))) == 2
    assert len(read_results(tmp_path)) == 4


def test_torn_result_line_is_ignored(tmp_path):
    (tmp_path / "x.jsonl").write_text(json.dumps({"run_id": "a", "n_success": 1}) + '\n{"run_id": "b", "n_su')
    assert len(read_results(tmp_path)) == 1


def test_wrong_result_count_raises(tmp_path):
    with pytest.raises(RuntimeError, match="expected 20"):
        run_jobs(_source(), _jobs("target")[:1], tmp_path, lambda: (lambda job: np.zeros(3, dtype=bool)))


def _save_policy(tmp_path, name, seed):
    from safetensors.torch import save_file

    path = tmp_path / name
    path.mkdir()
    state = {k: v.clone().contiguous() for k, v in make_tiny_policy(seed=seed).state_dict().items()}
    save_file(state, path / "model.safetensors")
    return path, state


def test_in_memory_merge_matches_and_writes_nothing(tmp_path):
    s1_dir, s1 = _save_policy(tmp_path, "s1", 0)
    ft_dir, ft = _save_policy(tmp_path, "ft", 1)
    before = sorted(p for p in tmp_path.rglob("*"))
    spec = AlphaSpec.parse("global=0.25").canonical
    # The runner uses the 16-layer grouping by default; the tiny model has 6 layers.
    state = build_state(_source(ckpt=ft_dir, s1_ckpt=s1_dir, alpha_spec=spec), num_layers=NUM_LAYERS)
    expected = merge_state_dicts(s1, ft, AlphaSpec.parse(spec), NUM_LAYERS)
    assert all(torch.equal(state[k], expected[k]) for k in expected)
    assert sorted(p for p in tmp_path.rglob("*")) == before


def test_no_merge_returns_the_checkpoint(tmp_path):
    ft_dir, ft = _save_policy(tmp_path, "ft", 1)
    state = build_state(_source(ckpt=ft_dir))
    assert all(torch.equal(state[k], ft[k]) for k in ft)


def test_merge_requires_the_reference(tmp_path):
    ft_dir, _ = _save_policy(tmp_path, "ft", 1)
    with pytest.raises(ValueError, match="s1_ckpt"):
        build_state(_source(ckpt=ft_dir, alpha_spec="global=0.50"))
