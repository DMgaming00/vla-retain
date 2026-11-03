import copy
import sys
from pathlib import Path

import pytest
import torch

from tests.tiny import NUM_LAYERS, make_batch, make_tiny_policy
from vlaretain.data.probes import (
    allocate_per_task,
    collate_frames,
    freeze_probe,
    load_probe,
    probe_manifest,
    raw_probe_batches,
    select_probe_episodes,
)
from vlaretain.data.self_replay import successful_frames, to_uint8_hwc
from vlaretain.data.splits import STAGE1_SUITES, STAGE2_SUITES, load_task_table
from vlaretain.drift.divergence import (
    DivergenceConfig,
    action_mmd,
    cka_per_layer,
    compute_probe_metrics,
    rbf_mmd2,
    shared_noise_action_distance,
    velocity_distance,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
CFG = DivergenceConfig(n_action_samples=4, n_velocity_draws=2)


@pytest.fixture(scope="module")
def pair():
    policy = make_tiny_policy(seed=0).eval()
    same = copy.deepcopy(policy).eval()
    other = copy.deepcopy(policy).eval()
    with torch.no_grad():
        for name, param in other.named_parameters():
            if "lm_expert" in name or "action_" in name:
                param.add_(0.05 * torch.randn_like(param))
    return policy, same, other


@pytest.fixture(scope="module")
def batches():
    return [make_batch(3, seed=1), make_batch(3, seed=2)]


def test_every_divergence_is_zero_when_policy_equals_s1(pair, batches):
    policy, same, _ = pair
    metrics = compute_probe_metrics(policy, same, batches, CFG)
    assert metrics["action_chunk_distance"] == 0.0
    assert metrics["velocity_distance"] == 0.0
    assert metrics["action_mmd"] == 0.0
    assert all(v == pytest.approx(0.0, abs=1e-9) for v in metrics["cka_distance"].values())
    assert sorted(metrics["cka_distance"]) == list(range(NUM_LAYERS))


def test_divergences_are_positive_for_a_different_policy(pair, batches):
    policy, _, other = pair
    assert shared_noise_action_distance(other, policy, batches, CFG) > 0
    assert velocity_distance(other, policy, batches, CFG) > 0
    assert action_mmd(other, policy, batches, CFG) > 0


def test_shared_noise_distance_is_deterministic(pair, batches):
    policy, _, other = pair
    a = shared_noise_action_distance(other, policy, batches, CFG)
    torch.manual_seed(999)  # global RNG state must not matter
    assert shared_noise_action_distance(other, policy, batches, CFG) == a
    assert shared_noise_action_distance(other, policy, batches, DivergenceConfig(seed=5)) != a


def test_velocity_distance_is_deterministic_and_ignores_padding(pair):
    policy, _, other = pair
    batch = make_batch(3, seed=1)
    a = velocity_distance(other, policy, [batch], CFG)
    assert velocity_distance(other, policy, [batch], CFG) == a
    batch_all_pad = dict(batch, action_is_pad=torch.ones_like(batch["action_is_pad"]))
    assert velocity_distance(other, policy, [batch_all_pad], CFG) == 0.0


def test_mmd_properties():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(16, 8, generator=g)
    y = torch.randn(16, 8, generator=g) + 3.0
    assert rbf_mmd2(x, x).item() == pytest.approx(0.0, abs=1e-12)
    assert rbf_mmd2(x, y).item() > rbf_mmd2(x, torch.randn(16, 8, generator=g)).item() > 0
    assert rbf_mmd2(x, y).item() == pytest.approx(rbf_mmd2(y, x).item(), rel=1e-9)
    for _ in range(20):
        assert rbf_mmd2(torch.randn(16, 4), torch.randn(16, 4)).item() >= 0


def test_cka_detects_lm_changes():
    policy = make_tiny_policy(seed=0).eval()
    other = copy.deepcopy(policy).eval()
    with torch.no_grad():
        for name, param in other.named_parameters():
            if "text_model.layers.1." in name:
                param.add_(0.5 * torch.randn_like(param))
    batch = [make_batch(8, seed=1)]
    result = cka_per_layer(other, policy, batch)
    assert result[0] == pytest.approx(0.0, abs=1e-9)  # upstream of the change
    assert result[NUM_LAYERS - 1] > 1e-4


# ---------- probes ----------
def _index_and_lengths():
    table = load_task_table()
    index, lengths, next_id = {}, {}, 0
    for suite, tasks in table.items():
        index[suite] = {}
        for task in tasks:
            index[suite][task["language"]] = list(range(next_id, next_id + 40))
            lengths.update({e: 100 + e % 7 for e in index[suite][task["language"]]})
            next_id += 40
    return index, lengths


def test_probe_allocation_is_balanced_and_exact():
    assert sum(allocate_per_task(10, 256)) == 256 and max(allocate_per_task(10, 256)) == 26
    assert sum(allocate_per_task(30, 256)) == 256


def test_probe_manifest_is_frozen_by_seed_and_uses_the_right_suites():
    index, lengths = _index_and_lengths()
    manifest = probe_manifest(index, lengths)
    assert manifest == probe_manifest(index, lengths)
    assert len(manifest["new"]) == 256 and len(manifest["old"]) == 256
    new_eps = {e for s in STAGE2_SUITES for v in index[s].values() for e in v}
    old_eps = {e for s in STAGE1_SUITES for v in index[s].values() for e in v}
    assert {e for e, _ in manifest["new"]} <= new_eps
    assert {e for e, _ in manifest["old"]} <= old_eps
    assert all(0 <= f < lengths[e] for e, f in manifest["new"] + manifest["old"])
    assert select_probe_episodes(index, lengths, STAGE2_SUITES, 256, 1) != manifest["new"]


def _items(n):
    return [
        {
            "observation.images.image": torch.rand(1, 3, 8, 8),
            "observation.state": torch.rand(1, 8),
            "action": torch.rand(50, 7),
            "action_is_pad": torch.zeros(50, dtype=torch.bool),
            "task": f"task {i}",
            "index": torch.tensor(i),
        }
        for i in range(n)
    ]


def test_probe_round_trip_and_refuses_to_overwrite(tmp_path):
    items = _items(5)
    new, old = collate_frames(items), collate_frames(_items(3))
    assert new["observation.images.image"].dtype == torch.uint8 and "index" not in new
    path = tmp_path / "probe.pt"
    manifest = {"version": "probe_v1", "new": [], "old": []}
    digest = freeze_probe(path, new, old, manifest)
    assert len(digest) == 16 and path.with_suffix(".json").exists()
    with pytest.raises(FileExistsError, match="frozen"):
        freeze_probe(path, new, old, manifest)
    loaded = load_probe(path)
    assert torch.equal(loaded["new"]["action"], new["action"]) and loaded["old"]["task"] == old["task"]
    pieces = raw_probe_batches(loaded["new"], 2)
    assert [len(b["task"]) for b in pieces] == [2, 2, 1]
    image = pieces[0]["observation.images.image"]
    assert image.dtype == torch.float32 and image.max() <= 1.0
    assert image[0].numpy() == pytest.approx(items[0]["observation.images.image"][0].numpy(), abs=1 / 254)


# ---------- drift rows and self-replay ----------
def test_metric_rows_cover_both_probe_splits_and_every_group():
    from drift import metric_rows
    from vlaretain.models.modules import GROUPS

    per_split = {"action_chunk_distance": 1.0, "velocity_distance": 2.0, "action_mmd": 3.0, "cka_distance": {0: 0.1, 1: 0.2}}
    rows = metric_rows(dict.fromkeys(GROUPS, 0.5), {"new": per_split, "old": per_split})
    assert len([r for r in rows if r["metric"] == "relative_l2"]) == len(GROUPS)
    for split in ("new", "old"):
        names = {r["metric"] for r in rows if r["probe_split"] == split}
        assert names == {"action_chunk_distance", "velocity_distance", "action_mmd", "cka_distance"}


def test_successful_frames_cut_at_first_success():
    data = {"success": torch.tensor([[0, 0, 1, 1], [0, 0, 0, 0]], dtype=torch.bool)}
    assert successful_frames(data, 0) == 3
    assert successful_frames(data, 1) is None


def test_uint8_hwc_conversion():
    assert to_uint8_hwc(torch.ones(3, 4, 5)).shape == (4, 5, 3)
