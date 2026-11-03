"""Checks on the real lerobot/smolvla_base. Skipped automatically if the model is not cached."""

import pytest
import torch
from huggingface_hub import try_to_load_from_cache

from vlaretain.config import MethodConfig, ModelConfig, RunConfig, TrainConfig
from vlaretain.methods import get_method
from vlaretain.models import load as model_load
from vlaretain.models.modules import GROUPS, group_counts

REPO = "lerobot/smolvla_base"
REVISION = ModelConfig().base_revision
VLM = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

_cached = (
    isinstance(try_to_load_from_cache(REPO, "model.safetensors", revision=REVISION), str)
    and isinstance(try_to_load_from_cache(VLM, "tokenizer.json"), str)
)
pytestmark = [pytest.mark.real_model, pytest.mark.skipif(not _cached, reason="smolvla_base is not cached")]

TOTAL_PARAMS = 450_046_176
VISION = 86_433_024
LM_HEAD = 47_308_800
EXPERT = 99_880_992
LM_KEY_IMAGES = model_load.LIBERO_IMAGE_KEYS


@pytest.fixture(scope="module")
def m0():
    return model_load.load_policy(REPO, revision=REVISION, device="cpu", train_expert_only=True)


def _batch(policy, n=2, seed=0):
    """A real-shaped LIBERO batch, passed through LeRobot's own SmolVLA preprocessor."""
    from lerobot.policies import make_pre_post_processors

    g = torch.Generator().manual_seed(seed)
    stats = {
        "observation.state": {"mean": torch.zeros(8), "std": torch.ones(8)},
        "action": {"mean": torch.zeros(7), "std": torch.ones(7)},
    }
    pre, _post = make_pre_post_processors(policy.config, dataset_stats=stats)
    raw = {k: torch.rand(n, 3, 256, 256, generator=g) for k in LM_KEY_IMAGES}
    raw["observation.state"] = torch.randn(n, 8, generator=g)
    raw["action"] = torch.randn(n, policy.config.chunk_size, 7, generator=g)
    raw["action_is_pad"] = torch.zeros(n, policy.config.chunk_size, dtype=torch.bool)
    raw["task"] = ["put the bowl on the plate", "turn on the stove"][:n]
    return pre(raw)


def _trainable(policy):
    return [(n, p) for n, p in policy.named_parameters() if p.requires_grad]


def test_every_parameter_is_float32_after_load(m0):
    assert {p.dtype for p in m0.parameters()} == {torch.float32}


def test_parameter_total_and_groups(m0):
    named = list(m0.named_parameters())
    assert sum(p.numel() for _, p in named) == TOTAL_PARAMS
    counts = group_counts(named, 16)
    assert set(counts) == set(GROUPS) and sum(counts.values()) == TOTAL_PARAMS
    assert counts["vision"] == VISION and counts["expert"] == EXPERT


def test_m0_trains_exactly_the_expert(m0):
    counts = group_counts(_trainable(m0), 16)
    assert counts["expert"] == EXPERT
    assert sum(v for k, v in counts.items() if k != "expert") == 0


def test_loss_runs_on_a_real_shaped_batch(m0):
    batch = _batch(m0)
    loss, _ = m0.forward(batch)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_sampling_with_supplied_noise_is_deterministic(m0):
    m0.config.num_steps = 2
    m0.eval()
    batch = _batch(m0)
    noise = torch.randn(2, m0.config.chunk_size, m0.config.max_action_dim, generator=torch.Generator().manual_seed(1))
    a = model_load.sample_action_chunk(m0, batch, noise)
    b = model_load.sample_action_chunk(m0, batch, noise)
    assert a.shape == (2, m0.config.chunk_size, 7)
    assert torch.equal(a, b)
    other = model_load.sample_action_chunk(m0, batch, noise + 1.0)
    assert not torch.equal(a, other)


def test_velocity_matches_the_training_loss(m0):
    """predict_velocity must agree with the model's own loss at the same (noise, time)."""
    m0.eval()
    batch = _batch(m0)
    g = torch.Generator().manual_seed(2)
    noise = torch.randn(2, m0.config.chunk_size, m0.config.max_action_dim, generator=g)
    time = torch.tensor([0.3, 0.7])
    with torch.no_grad():
        loss, _ = m0.forward(dict(batch), noise=noise, time=time)
        actions = m0.prepare_action(batch)
        x_t = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
        velocity = model_load.predict_velocity(m0, batch, x_t, time)
    ours = ((noise - actions - velocity)[:, :, :7] ** 2).mean()
    assert ours.item() == pytest.approx(loss.item(), rel=1e-4)


def test_per_layer_hidden_states_can_be_read(m0):
    batch = _batch(m0)
    with torch.no_grad():
        hidden, pad_mask = model_load.lm_hidden_states(m0, batch)
    assert sorted(hidden) == list(range(16))
    for tensor in hidden.values():
        assert tensor.shape[0] == 2 and tensor.shape[1] == pad_mask.shape[1] and tensor.shape[2] == 960
    assert not torch.equal(hidden[0], hidden[15])
    assert model_load.pool_hidden(hidden[7], pad_mask).shape == (2, 960)


def _report(capsys, label, policy):
    counts = group_counts(_trainable(policy), 16)
    with capsys.disabled():
        print(f"\n[{label}] trainable={sum(counts.values()):,} {counts}")
    return counts


def test_m1_trainable_set(capsys):
    policy = model_load.load_policy(REPO, revision=REVISION, device="cpu", train_expert_only=False)
    counts = _report(capsys, "M1", policy)
    assert counts["vision"] == 0
    assert counts["expert"] == EXPERT
    assert counts["connector"] == 11_796_480
    assert all(counts[g] > 0 for g in ("lm_early", "lm_mid", "lm_late"))
    names = [n for n, _ in _trainable(policy)]
    # LeRobot always keeps the last LM layer, the LM norm, and lm_head frozen.
    assert not any("text_model.layers.15." in n or "text_model.norm" in n or "lm_head" in n for n in names)
    assert any("text_model.layers.14." in n for n in names)


def test_m2_trainable_set(capsys, monkeypatch):
    cfg = RunConfig(
        method=MethodConfig(name="lora", params={"r": 16, "alpha": 32, "dropout": 0.05}),
        train=TrainConfig(device="cpu"),
    )

    def load(s1_ckpt, cfg, *, train_expert_only=None):
        return model_load.load_policy(REPO, revision=REVISION, device="cpu", train_expert_only=True)

    monkeypatch.setattr(model_load, "load_policy_for_training", load)
    method = get_method(cfg)
    policy = method.build_model(None, cfg)
    counts = _report(capsys, "M2", policy)
    names = [n for n, _ in _trainable(policy)]
    assert all("lora_A" in n or "lora_B" in n for n in names)
    assert counts["vision"] == 0 and counts["connector"] == 0
    assert all(counts[g] > 0 for g in ("lm_early", "lm_mid", "lm_late", "expert"))
    for suffix in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
        assert any(suffix in n and "text_model" in n for n in names), suffix
        assert any(suffix in n and "lm_expert" in n for n in names), suffix
    assert method.check_trainable(policy)
    batch = _batch(policy)
    loss, _ = policy.forward(batch)
    loss.backward()
    grads = [p.grad for n, p in policy.named_parameters() if "lora_B" in n and "lm_expert" in n]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)
