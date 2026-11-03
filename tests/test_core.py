import numpy as np
import pytest
import torch
from scipy import stats as sps

from tests.tiny import NUM_LAYERS, make_tiny_policy
from vlaretain.config import load_config, run_id_of
from vlaretain.data.mix import MixedBatchSampler
from vlaretain.data.splits import build_episode_index, language_to_suite, load_task_table, replay_pool
from vlaretain.drift.cka import linear_cka
from vlaretain.drift.params import relative_l2_per_group
from vlaretain.eval.stats import paired_bootstrap, wilson_interval
from vlaretain.models.modules import GROUPS, group_counts, lm_layer_group, module_group


# ---------- config ----------
def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_config_merge_override_and_hash(tmp_path):
    a = _write(tmp_path, "a.yaml", "seed: 1\ntrain:\n  lr: 1e-4\n  steps: 10\n")
    b = _write(tmp_path, "b.yaml", "train:\n  steps: 20\nmethod:\n  name: l2sp\n  params: {lam: 1e-2}\n")
    cfg = load_config([a, b], ["train.lr=3e-5"])
    assert cfg.train.lr == pytest.approx(3e-5) and isinstance(cfg.train.lr, float)
    assert cfg.train.steps == 20 and cfg.method.name == "l2sp"
    assert run_id_of(cfg) == run_id_of(load_config([a, b], ["train.lr=3e-5"]))
    assert len(run_id_of(cfg)) == 10


def test_run_id_ignores_paths_but_not_semantics(tmp_path):
    a = _write(tmp_path, "a.yaml", "seed: 1\n")
    base = run_id_of(load_config([a]))
    assert run_id_of(load_config([a], ["paths.run_root=/elsewhere", "data.num_workers=2"])) == base
    assert run_id_of(load_config([a], ["seed=2"])) != base
    assert run_id_of(load_config([a], ["train.lr=1e-3"])) != base


def test_unknown_config_key_raises(tmp_path):
    a = _write(tmp_path, "a.yaml", "train:\n  learning_rate: 1\n")
    with pytest.raises(ValueError, match="Unknown config keys"):
        load_config([a])


# ---------- splits ----------
def _fake_index(per_task=50):
    table = load_task_table()
    index, next_id = {}, 0
    for suite, tasks in table.items():
        index[suite] = {}
        for task in tasks:
            index[suite][task["language"]] = list(range(next_id, next_id + per_task))
            next_id += per_task
    return index


def test_task_table_shape():
    table = load_task_table()
    assert {s: len(t) for s, t in table.items()} == {
        "libero_spatial": 10, "libero_object": 10, "libero_goal": 10, "libero_10": 10,
    }  # fmt: skip
    assert len(language_to_suite()) == 40


def test_build_episode_index_groups_and_rejects_unknown():
    language = next(iter(language_to_suite()))
    index = build_episode_index({3: language, 1: language})
    suite = language_to_suite()[language][0]
    assert index[suite][language] == [1, 3]
    with pytest.raises(KeyError):
        build_episode_index({0: "fly to the moon"})


@pytest.mark.parametrize("fraction", [0.02, 0.10])
def test_replay_pool_size_determinism_and_membership(fraction):
    index = _fake_index()
    pool = replay_pool(index, fraction, seed=7)
    assert len(pool) == round(fraction * 1500)
    assert pool == replay_pool(index, fraction, seed=7)
    assert pool != replay_pool(index, fraction, seed=8)
    stage1 = {e for s in ("libero_spatial", "libero_object", "libero_goal") for v in index[s].values() for e in v}
    assert set(pool) <= stage1 and len(set(pool)) == len(pool)


def test_replay_pool_spreads_across_tasks():
    index = _fake_index()
    pool = set(replay_pool(index, 0.10, seed=0))
    for suite in ("libero_spatial", "libero_object", "libero_goal"):
        for episodes in index[suite].values():
            assert len(pool & set(episodes)) == 5


# ---------- batch mixing ----------
def test_mix_ratio_is_exact_in_the_long_run():
    sampler = MixedBatchSampler(1000, 100, 32, 0.2, seed=0, total_steps=500)
    batches = list(sampler)
    assert all(len(b) == 32 for b in batches)
    replay = sum(i >= 1000 for b in batches for i in b)
    assert replay == pytest.approx(0.2 * 32 * 500, abs=1)
    assert {sum(i >= 1000 for i in b) for b in batches} == {6, 7}


def test_mix_is_deterministic_and_stateless_on_resume():
    full = list(MixedBatchSampler(100, 10, 8, 0.2, seed=3, total_steps=50))
    resumed = list(MixedBatchSampler(100, 10, 8, 0.2, seed=3, total_steps=50, start_step=20))
    assert full[20:] == resumed
    assert full != list(MixedBatchSampler(100, 10, 8, 0.2, seed=4, total_steps=50))


def test_mix_covers_each_pool_without_replacement_per_epoch():
    batches = list(MixedBatchSampler(64, 16, 8, 0.2, seed=0, total_steps=8))
    flat = [i for b in batches for i in b]
    target = [i for i in flat if i < 64]
    assert len(target) == len(set(target))


def test_mix_without_replay():
    batches = list(MixedBatchSampler(100, 0, 10, 0.0, seed=0, total_steps=10))
    assert sorted(i for b in batches for i in b) == list(range(100))
    with pytest.raises(ValueError):
        MixedBatchSampler(100, 0, 10, 0.2, seed=0, total_steps=1)


# ---------- module groups ----------
def test_every_parameter_maps_to_exactly_one_group():
    policy = make_tiny_policy()
    named = list(policy.named_parameters())
    assert named
    groups = [module_group(n, NUM_LAYERS) for n, _ in named]
    assert set(groups) == set(GROUPS)
    counts = group_counts(named, NUM_LAYERS)
    assert sum(counts.values()) == sum(p.numel() for _, p in named)


def test_unknown_parameter_raises():
    with pytest.raises(KeyError):
        module_group("model.something_new.weight")


def test_layer_groups_for_sixteen_layers():
    groups = [lm_layer_group(i, 16) for i in range(16)]
    assert groups == ["lm_early"] * 5 + ["lm_mid"] * 6 + ["lm_late"] * 5


# ---------- statistics ----------
@pytest.mark.parametrize("k,n", [(0, 10), (10, 10), (7, 20), (150, 300), (1, 700)])
def test_wilson_matches_scipy(k, n):
    ci = sps.binomtest(k, n).proportion_ci(confidence_level=0.95, method="wilson")
    lo, hi = wilson_interval(k, n)
    assert lo == pytest.approx(ci.low, abs=1e-9) and hi == pytest.approx(ci.high, abs=1e-9)


def test_wilson_edge_cases():
    assert wilson_interval(0, 0) == (None, None)
    with pytest.raises(ValueError):
        wilson_interval(5, 3)


def test_paired_bootstrap():
    a = np.array([0.9, 0.8, 0.7, 0.95, 0.85])
    result = paired_bootstrap(a, a - 0.1, n_boot=500)
    assert result["mean_diff"] == pytest.approx(0.1)
    assert result["ci_lo"] == pytest.approx(0.1) and result["ci_hi"] == pytest.approx(0.1)
    rng = np.random.default_rng(0)
    b = rng.normal(0, 1, 200)
    wide = paired_bootstrap(b + 0.5, b + rng.normal(0, 1, 200), n_boot=2000, seed=1)
    assert wide["ci_lo"] < wide["mean_diff"] < wide["ci_hi"]
    assert paired_bootstrap(a, a - 0.1, n_boot=100, seed=5) == paired_bootstrap(a, a - 0.1, n_boot=100, seed=5)


# ---------- weight drift ----------
def test_relative_l2_known_value_and_zero_at_self():
    policy = make_tiny_policy()
    ref = {k: v.clone() for k, v in policy.state_dict().items()}
    assert all(v == 0 for v in relative_l2_per_group(ref, ref, NUM_LAYERS).values())
    moved = {k: v.clone() for k, v in ref.items()}
    for key in moved:
        if "connector" in key and torch.is_floating_point(moved[key]):
            moved[key] = moved[key] * 1.5
    drift = relative_l2_per_group(moved, ref, NUM_LAYERS)
    assert drift["connector"] == pytest.approx(0.5, rel=1e-6)
    assert drift["expert"] == 0


# ---------- CKA ----------
def _reference_cka(x: np.ndarray, y: np.ndarray) -> float:
    """HSIC in Gram-matrix form, as in Kornblith et al. (2019), equation 1 with linear kernels."""
    n = x.shape[0]
    centering = np.eye(n) - np.ones((n, n)) / n
    kx, ky = centering @ (x @ x.T) @ centering, centering @ (y @ y.T) @ centering
    return float(np.sum(kx * ky) / np.sqrt(np.sum(kx * kx) * np.sum(ky * ky)))


def test_cka_matches_reference_implementation():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 12))
    y = 0.5 * x[:, :8] + rng.normal(size=(50, 8))
    assert linear_cka(torch.tensor(x), torch.tensor(y)) == pytest.approx(_reference_cka(x, y), abs=1e-10)


def test_cka_invariances():
    rng = np.random.default_rng(1)
    x = torch.tensor(rng.normal(size=(40, 10)))
    y = torch.tensor(rng.normal(size=(40, 10)))
    q, _ = torch.linalg.qr(torch.tensor(rng.normal(size=(10, 10))))
    base = linear_cka(x, y)
    assert linear_cka(x @ q, y) == pytest.approx(base, abs=1e-10)
    assert linear_cka(x * 7.5, y) == pytest.approx(base, abs=1e-10)
    assert linear_cka(x, x) == pytest.approx(1.0, abs=1e-10)
    assert linear_cka(x @ q * 3.0, x) == pytest.approx(1.0, abs=1e-10)


def test_cka_rejects_mismatched_examples():
    with pytest.raises(ValueError):
        linear_cka(torch.zeros(4, 2), torch.zeros(5, 2))


def test_bfloat16_cannot_hold_small_updates():
    """The reason we upcast the checkpoint to float32 before training."""
    weight = torch.tensor(0.02, dtype=torch.bfloat16)
    update = torch.tensor(3e-5, dtype=torch.bfloat16)
    assert weight - update == weight
    assert torch.tensor(0.02, dtype=torch.float32) - 3e-5 != torch.tensor(0.02, dtype=torch.float32)
