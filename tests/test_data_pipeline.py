"""Our data code against a real LeRobotDataset written to disk, without any Hub access."""

import pytest
import torch
from lerobot.datasets import LeRobotDataset
from torch.utils.data import ConcatDataset, DataLoader

from tests.tiny import make_tiny_policy
from vlaretain.config import DataConfig, RunConfig
from vlaretain.data.mix import MixedBatchSampler
from vlaretain.data.probes import collate_frames, raw_probe_batches
from vlaretain.train.loop import _subset_dataset

REPO = "local/tiny"
LENGTHS = [8, 6, 10, 7]


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    path = tmp_path_factory.mktemp("ds") / "tiny"
    features = {
        "observation.images.image": {"dtype": "image", "shape": (8, 8, 3), "names": ["height", "width", "channel"]},
        "observation.images.image2": {"dtype": "image", "shape": (8, 8, 3), "names": ["height", "width", "channel"]},
        "observation.state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
        "action": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
    }
    ds = LeRobotDataset.create(repo_id=REPO, fps=10, features=features, root=path, use_videos=False)
    for episode, length in enumerate(LENGTHS):
        for t in range(length):
            # The action encodes (episode, frame), so tests can verify which frame came back.
            action = torch.full((7,), float(episode * 100 + t)).numpy()
            ds.add_frame({
                "observation.images.image": torch.zeros(8, 8, 3, dtype=torch.uint8).numpy(),
                "observation.images.image2": torch.zeros(8, 8, 3, dtype=torch.uint8).numpy(),
                "observation.state": torch.full((8,), float(episode)).numpy(),
                "action": action,
                "task": f"task {episode % 2}",
            })  # fmt: skip
        ds.save_episode()
    ds.finalize()
    return path


def _subset(root, episodes):
    cfg = RunConfig(data=DataConfig(repo_id=REPO, revision=None))
    return _subset_dataset(cfg, make_tiny_policy().config, episodes, repo_id=REPO, root=root)


def test_episode_subset_returns_only_those_frames(root):
    ds = _subset(root, [1, 3])
    assert len(ds) == LENGTHS[1] + LENGTHS[3]
    assert {int(ds[i]["episode_index"]) for i in range(len(ds))} == {1, 3}


def test_item_shapes_and_action_chunk(root):
    ds = _subset(root, [0])
    item = ds[2]
    chunk = make_tiny_policy().config.chunk_size
    assert item["action"].shape == (chunk, 7)
    assert item["action"][0, 0] == 2 and item["action"][1, 0] == 3
    assert item["task"] == "task 0"
    assert item["observation.images.image"].shape[-3:] == (3, 8, 8)


def test_action_chunk_stops_at_the_episode_end_and_is_flagged(root):
    ds = _subset(root, [0])
    item = ds[LENGTHS[0] - 1]
    assert item["action_is_pad"][0].item() is False and item["action_is_pad"][1:].all()


def test_concat_with_mixed_sampler_gives_the_right_share_of_replay(root):
    target, replay = _subset(root, [0, 1, 2]), _subset(root, [3])
    dataset = ConcatDataset([target, replay])
    sampler = MixedBatchSampler(len(target), len(replay), 5, 0.2, seed=0, total_steps=20)
    loader = DataLoader(dataset, batch_sampler=sampler)
    episodes = torch.cat([b["episode_index"].flatten() for b in loader])
    assert (episodes == 3).float().mean().item() == pytest.approx(0.2, abs=1e-6)


def test_probe_collation_on_real_items(root):
    ds = _subset(root, [2])
    split = collate_frames([ds[i] for i in range(4)])
    assert split["observation.images.image"].dtype == torch.uint8
    assert split["observation.images.image"].shape == (4, 3, 8, 8)
    assert split["observation.state"].shape == (4, 8) and split["action"].shape[0] == 4
    assert len(split["task"]) == 4 and "action_is_pad" in split
    assert len(raw_probe_batches(split, 3)) == 2
