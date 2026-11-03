import json
import os
import signal

import pytest
import torch
from torch.utils.data import Dataset

from tests.tiny import NUM_LAYERS, make_batch, make_tiny_policy
from vlaretain.config import DataConfig, MethodConfig, ModelConfig, PathsConfig, RunConfig, TrainConfig
from vlaretain.data.mix import MixedBatchSampler
from vlaretain.methods import get_method
from vlaretain.methods.anchor import FeatureAnchor
from vlaretain.methods.l2sp import L2SP, l2sp_penalty
from vlaretain.models import load as model_load
from vlaretain.models.modules import group_counts
from vlaretain.train.loop import TrainData, find_resume_dir, lr_lambda, train


class TinyFrames(Dataset):
    """Tensors with the shapes the preprocessor would produce. Target and replay frames differ, so tests can tell them apart."""

    def __init__(self, n_target=40, n_replay=10):
        batch = make_batch(n_target + n_replay, seed=11)
        self.items = batch
        self.n = n_target + n_replay

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {k: v[i] for k, v in self.items.items()}


def tiny_data(n_replay=10):
    return TrainData(TinyFrames(40, n_replay), 40, n_replay, lambda b: b)


METHODS = {
    "sft_expert": ({}, {}),
    "sft_full": ({}, {}),
    "lora": ({"r": 2, "alpha": 4, "dropout": 0.05}, {}),
    "replay": ({}, {"replay_fraction": 0.2, "replay_pool_fraction": 0.1}),
    "self_replay": ({}, {"replay_fraction": 0.2, "replay_source": "self_rollouts"}),
    "merge": ({}, {}),
    "layer_merge": ({}, {}),
    "l2sp": ({"lam": 0.1}, {}),
    "anchor": ({"lam": 0.1}, {}),
    "stage1": ({}, {}),
}


def tiny_cfg(tmp_path, name, steps=20, **train_kwargs) -> RunConfig:
    params, data = METHODS[name]
    return RunConfig(
        stage=1 if name == "stage1" else 2,
        seed=3,
        model=ModelConfig(train_expert_only=True),
        data=DataConfig(num_workers=0, **data),
        train=TrainConfig(
            steps=steps, batch_size=8, lr=1e-3, min_lr=1e-4, warmup_steps=2, save_every=5,
            milestones=[10, 20], log_every=1, device="cpu", **train_kwargs,
        ),
        method=MethodConfig(name=name, params=params),
        paths=PathsConfig(run_root=str(tmp_path / "runs")),
    )  # fmt: skip


@pytest.fixture(autouse=True)
def tiny_loader(monkeypatch):
    """Keep the real policy class and freezing logic, on a tiny model loaded offline."""

    def load(s1_ckpt, cfg, *, train_expert_only=None):
        flag = cfg.model.train_expert_only if train_expert_only is None else train_expert_only
        policy = make_tiny_policy(seed=0, train_expert_only=flag)
        policy.train()
        return policy

    monkeypatch.setattr(model_load, "load_policy_for_training", load)


def _losses(cfg):
    rows = [json.loads(line) for line in (cfg.run_dir / "train_log.jsonl").read_text().splitlines()]
    return {r["step"]: r["loss"] for r in rows}


@pytest.mark.parametrize("name", list(METHODS))
def test_method_runs_twenty_steps_and_marks_done(tmp_path, name):
    cfg = tiny_cfg(tmp_path, name)
    result = train(cfg, get_method(cfg), tiny_data())
    assert result.status == "finished" and result.step == 20
    assert (cfg.run_dir / "DONE").exists()
    assert (cfg.run_dir / "run_meta.json").exists()
    for step in (10, 20):
        assert (cfg.run_dir / "checkpoints" / f"step_{step:06d}").is_dir()
    losses = _losses(cfg)
    assert len(losses) == 20 and all(torch.isfinite(torch.tensor(v)) for v in losses.values())


@pytest.mark.parametrize("name", ["sft_full", "lora", "replay", "l2sp", "anchor"])
def test_resume_at_step_ten_reproduces_step_twenty_loss(tmp_path, name):
    straight = tiny_cfg(tmp_path / "a", name)
    train(straight, get_method(straight), tiny_data())

    resumed = tiny_cfg(tmp_path / "b", name)
    first = train(resumed, get_method(resumed), tiny_data(), stop_after=10)
    assert first.status == "stopped" and not (resumed.run_dir / "DONE").exists()
    second = train(resumed, get_method(resumed), tiny_data())
    assert second.status == "finished"

    assert _losses(resumed)[20] == pytest.approx(_losses(straight)[20], rel=1e-5, abs=1e-6)


def test_finished_run_is_not_retrained(tmp_path):
    cfg = tiny_cfg(tmp_path, "sft_expert")
    train(cfg, get_method(cfg), tiny_data())
    assert train(cfg, get_method(cfg), tiny_data()).status == "already_done"


@pytest.mark.parametrize("sig", [signal.SIGUSR1, signal.SIGTERM])
def test_signal_saves_and_exits_then_resumes(tmp_path, sig):
    cfg = tiny_cfg(tmp_path, "sft_expert")

    def fire(step):
        if step == 7:
            os.kill(os.getpid(), sig)

    result = train(cfg, get_method(cfg), tiny_data(), on_step=fire)
    assert result.status == "preempted" and result.step == 7
    assert torch.load(find_resume_dir(cfg.run_dir / "checkpoints") / "training_state.pt", weights_only=False)["step"] == 7
    assert train(cfg, get_method(cfg), tiny_data()).status == "finished"


def test_gpu_hours_survive_resume(tmp_path):
    cfg = tiny_cfg(tmp_path, "sft_expert")
    first = train(cfg, get_method(cfg), tiny_data(), stop_after=5)
    second = train(cfg, get_method(cfg), tiny_data())
    assert second.gpu_hours >= first.gpu_hours > 0


def test_killed_mid_save_falls_back_to_previous_checkpoint(tmp_path):
    cfg = tiny_cfg(tmp_path, "sft_expert")
    train(cfg, get_method(cfg), tiny_data(), stop_after=5)
    root = cfg.run_dir / "checkpoints"
    (root / "latest").rename(root / "latest.old")
    (root / "latest.tmp").mkdir()
    assert find_resume_dir(root) == root / "latest.old"
    assert train(cfg, get_method(cfg), tiny_data()).status == "finished"


# ---------- method-specific ----------
def test_l2sp_penalty_is_zero_at_init_and_grows(tmp_path):
    cfg = tiny_cfg(tmp_path, "l2sp")
    method = L2SP(cfg)
    policy = method.build_model(None, cfg)
    assert l2sp_penalty(policy, method.reference).item() == 0.0
    with torch.no_grad():
        for param in policy.parameters():
            if param.requires_grad:
                param.add_(2.0)
    assert l2sp_penalty(policy, method.reference).item() == pytest.approx(4.0)


def test_l2sp_reference_stays_at_s1_after_resume(tmp_path):
    cfg = tiny_cfg(tmp_path, "l2sp")
    train(cfg, L2SP(cfg), tiny_data(), stop_after=10)
    method = L2SP(cfg)
    policy = method.build_model(None, cfg)
    method.load_weights(policy, cfg.run_dir / "checkpoints" / "latest")
    assert l2sp_penalty(policy, method.reference).item() > 0


def test_anchor_distance_is_zero_at_init_and_gradients_flow(tmp_path):
    cfg = tiny_cfg(tmp_path, "anchor")
    method = FeatureAnchor(cfg)
    policy = method.build_model(None, cfg)
    loss, info = method.extra_loss(policy, make_batch(), 0)
    assert info["anchor_distance"] == pytest.approx(0.0, abs=1e-5)
    with torch.no_grad():
        for name, param in policy.named_parameters():
            if "text_model.layers.0." in name and param.requires_grad:
                param.add_(0.3 * torch.randn_like(param))
    loss, info = method.extra_loss(policy, make_batch(), 0)
    assert info["anchor_distance"] > 1e-4
    loss.backward()
    grads = [p.grad for n, p in policy.named_parameters() if "text_model.layers.0." in n and p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)
    assert all(p.grad is None for p in method.reference.parameters())


def test_replay_ratio_reaches_the_model(tmp_path):
    """Check the real loader path: replay frames should be about 20% of what the loop would consume."""
    sampler = MixedBatchSampler(40, 10, 8, 0.2, seed=3, total_steps=100)
    frames = [i for b in sampler for i in b]
    assert sum(i >= 40 for i in frames) / len(frames) == pytest.approx(0.2, abs=0.005)


def test_replay_requires_a_fraction(tmp_path):
    cfg = tiny_cfg(tmp_path, "replay")
    cfg.data.replay_fraction = 0.0
    with pytest.raises(ValueError, match="replay_fraction"):
        get_method(cfg)


def test_config_meant_to_train_the_lm_fails_with_zero_trainable_lm(tmp_path):
    cfg = tiny_cfg(tmp_path, "sft_full")
    method = get_method(cfg)
    policy = make_tiny_policy(train_expert_only=True)  # the misconfiguration this check exists to catch
    with pytest.raises(RuntimeError, match="0 trainable parameters"):
        method.check_trainable(policy)


def test_expert_only_config_fails_if_the_lm_is_trainable(tmp_path):
    cfg = tiny_cfg(tmp_path, "sft_expert")
    policy = make_tiny_policy(train_expert_only=False)
    with pytest.raises(RuntimeError, match="should be frozen"):
        get_method(cfg).check_trainable(policy)


def test_trainable_sets_on_the_real_policy_class(tmp_path):
    named = lambda p: [(n, x) for n, x in p.named_parameters() if x.requires_grad]  # noqa: E731
    m0 = group_counts(named(make_tiny_policy(train_expert_only=True)), NUM_LAYERS)
    assert m0["expert"] > 0 and sum(v for k, v in m0.items() if k != "expert") == 0
    m1 = group_counts(named(make_tiny_policy(train_expert_only=False)), NUM_LAYERS)
    assert m1["vision"] == 0
    assert all(m1[g] > 0 for g in ("connector", "lm_early", "lm_mid", "lm_late", "expert"))

    cfg = tiny_cfg(tmp_path, "lora")
    method = get_method(cfg)
    policy = method.build_model(None, cfg)
    trainable = [n for n, _ in named(policy)]
    assert trainable and all("lora_" in n for n in trainable)
    assert any("lm_expert" in n for n in trainable) and any("text_model" in n for n in trainable)
    assert any("mlp.gate_proj" in n for n in trainable) and any("self_attn.o_proj" in n for n in trainable)
    assert method.check_trainable(policy)["expert"] > 0


def test_loop_uses_only_the_interface():
    """The loop source must not name any method."""
    import inspect

    from vlaretain import train as train_pkg  # noqa: F401
    from vlaretain.train import loop

    source = inspect.getsource(loop.train)
    for name in METHODS:
        assert f'"{name}"' not in source and f"'{name}'" not in source
    assert "cfg.method.name" not in source


def test_lr_schedule_shape():
    kw = dict(peak=1e-3, floor=1e-4, warmup=10, total=110)
    assert lr_lambda(0, **kw) == pytest.approx(0.1)
    assert lr_lambda(9, **kw) == pytest.approx(1.0)
    assert lr_lambda(10, **kw) == pytest.approx(1.0)
    assert lr_lambda(60, **kw) == pytest.approx(0.55)
    assert lr_lambda(110, **kw) == pytest.approx(0.1)




def test_resume_comparison_is_sensitive_to_the_seed(tmp_path):
    """Guards the resume tests: a different seed must change the step-20 loss, or they prove nothing."""
    a = tiny_cfg(tmp_path / "a", "sft_full")
    b = tiny_cfg(tmp_path / "b", "sft_full")
    b.seed = 4
    train(a, get_method(a), tiny_data())
    train(b, get_method(b), tiny_data())
    assert _losses(a)[20] != pytest.approx(_losses(b)[20], rel=1e-5)
