"""Loading SmolVLA checkpoints, plus the model operations LeRobot does not expose as one call."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import torch
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.common.vla_utils import make_att_2d_masks
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

LIBERO_IMAGE_KEYS = ("observation.images.image", "observation.images.image2")
LIBERO_STATE_DIM = 8
LIBERO_ACTION_DIM = 7
LIBERO_IMAGE_SHAPE = (3, 256, 256)

CKPT_META = "vlaretain_ckpt.json"


def libero_features() -> tuple[dict[str, PolicyFeature], dict[str, PolicyFeature]]:
    inputs = {k: PolicyFeature(type=FeatureType.VISUAL, shape=LIBERO_IMAGE_SHAPE) for k in LIBERO_IMAGE_KEYS}
    inputs[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(LIBERO_STATE_DIM,))
    outputs = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(LIBERO_ACTION_DIM,))}
    return inputs, outputs


def apply_freezing(policy: torch.nn.Module) -> None:
    """Re-apply LeRobot's freezing rules, using the flags currently on the policy config."""
    vlm_with_expert = policy.model.vlm_with_expert
    vlm_with_expert.freeze_vision_encoder = policy.config.freeze_vision_encoder
    vlm_with_expert.train_expert_only = policy.config.train_expert_only
    vlm_with_expert.set_requires_grad()
    policy.model.set_requires_grad()


def load_policy(
    pretrained: str | Path, *, revision: str | None = None, device: str = "cpu", set_libero_features: bool = True,
    train_expert_only: bool | None = None, freeze_vision_encoder: bool | None = None,
    train_state_proj: bool | None = None,
) -> torch.nn.Module:  # fmt: skip
    """Load a full checkpoint in float32.

    The base checkpoint stores most weights in bfloat16, where our learning rates round to no update at all,
    so we upcast everything. `load_vlm_weights` is off because the checkpoint already holds the VLM.
    """
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    config = PreTrainedConfig.from_pretrained(pretrained, revision=revision)
    config.load_vlm_weights = False
    config.device = device
    if set_libero_features:
        config.input_features, config.output_features = libero_features()
    if train_expert_only is not None:
        config.train_expert_only = train_expert_only
    if freeze_vision_encoder is not None:
        config.freeze_vision_encoder = freeze_vision_encoder
    if train_state_proj is not None:
        config.train_state_proj = train_state_proj
    policy = SmolVLAPolicy.from_pretrained(pretrained, config=config, revision=revision)
    policy = policy.to(dtype=torch.float32)
    apply_freezing(policy)
    return policy


def load_policy_for_training(s1_ckpt, cfg, *, train_expert_only: bool | None = None) -> torch.nn.Module:
    """Start from S1, or from the base model during Stage-1, with this config's trainable set."""
    source, revision = (s1_ckpt, None) if s1_ckpt is not None else (cfg.model.base, cfg.model.base_revision)
    policy = load_policy(
        str(source),
        revision=revision,
        device=cfg.train.device,
        train_expert_only=cfg.model.train_expert_only if train_expert_only is None else train_expert_only,
        freeze_vision_encoder=cfg.model.freeze_vision_encoder,
        train_state_proj=cfg.model.train_state_proj,
    )
    policy.train()
    return policy


def load_state_for_eval(ckpt_dir: str | Path, s1_ckpt: str | Path | None = None) -> dict[str, torch.Tensor]:
    """Read a checkpoint's full state dict on CPU. Adapter-only checkpoints are baked onto their S1 base."""
    from safetensors.torch import load_file

    from vlaretain.methods.merge import lora_merged_state_dict

    ckpt_dir = Path(ckpt_dir)
    meta_path = ckpt_dir / CKPT_META
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"kind": "full"}
    if meta["kind"] == "full":
        return {k: v.to(torch.float32) for k, v in load_file(ckpt_dir / "model.safetensors").items()}
    base_dir = Path(s1_ckpt) if s1_ckpt is not None else Path(meta["base"])
    base = {k: v.to(torch.float32) for k, v in load_file(base_dir / "model.safetensors").items()}
    adapter = load_file(ckpt_dir / "adapter_model.safetensors")
    return lora_merged_state_dict(base, adapter, scale=float(meta["lora_alpha"]) / float(meta["lora_r"]))


def _to_model_inputs(policy, batch: dict) -> tuple:
    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    return images, img_masks, batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK], state


def predict_velocity(policy, batch: dict, x_t: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    """v_theta(x_t, t | observation) for padded `x_t` of shape (B, chunk, max_action_dim).

    Follows VLAFlowMatching.forward, which computes the same value but only returns the loss.
    """
    model = policy.model
    images, img_masks, tokens, masks, state = _to_model_inputs(policy, batch)
    prefix_embs, prefix_pad, prefix_att = model.embed_prefix(images, img_masks, tokens, masks, state=state)
    suffix_embs, suffix_pad, suffix_att = model.embed_suffix(x_t, time)
    pad_masks = torch.cat([prefix_pad, suffix_pad], dim=1)
    att_masks = torch.cat([prefix_att, suffix_att], dim=1)
    att_2d = make_att_2d_masks(pad_masks, att_masks)
    position_ids = torch.cumsum(pad_masks, dim=1) - 1
    (_, suffix_out), _ = model.vlm_with_expert.forward(
        attention_mask=att_2d,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, suffix_embs],
        use_cache=False,
    )
    suffix_out = suffix_out[:, -policy.config.chunk_size :].to(dtype=torch.float32)
    return model.action_out_proj(suffix_out)


def sample_action_chunk(policy, batch: dict, noise: torch.Tensor) -> torch.Tensor:
    """Sample an action chunk in normalized space with caller-supplied noise of padded shape."""
    return policy.predict_action_chunk(dict(batch), noise=noise)


@contextmanager
def capture_lm_hidden(policy) -> Iterator[Callable[[], dict[int, torch.Tensor]]]:
    """Capture the LM's residual stream after each layer, during any forward inside the context.

    The VLM loop never calls `layer.forward`, so we hook what it does call: the stream entering layer i is the input
    of its `input_layernorm`, and the stream leaving the last layer is the input of `text_model.norm`.
    """
    text_model = policy.model.vlm_with_expert.get_vlm_model().text_model
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer_out: int):
        def hook(_module, args):
            captured[layer_out] = args[0]

        return hook

    n = len(text_model.layers)
    for i in range(1, n):
        handles.append(text_model.layers[i].input_layernorm.register_forward_pre_hook(make_hook(i - 1)))
    handles.append(text_model.norm.register_forward_pre_hook(make_hook(n - 1)))
    try:
        yield lambda: captured
    finally:
        for handle in handles:
            handle.remove()


def lm_hidden_states(policy, batch: dict) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    """Run the prefix through the LM and return ({layer: (B, L, D)}, pad mask (B, L)). Keeps gradients if enabled."""
    model = policy.model
    images, img_masks, tokens, masks, state = _to_model_inputs(policy, batch)
    prefix_embs, prefix_pad, prefix_att = model.embed_prefix(images, img_masks, tokens, masks, state=state)
    att_2d = make_att_2d_masks(prefix_pad, prefix_att)
    position_ids = torch.cumsum(prefix_pad, dim=1) - 1
    with capture_lm_hidden(policy) as get:
        # use_cache=True selects the VLM-only path; see docs/lerobot_notes.md.
        model.vlm_with_expert.forward(
            attention_mask=att_2d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        return dict(get()), prefix_pad


def pool_hidden(hidden: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
    """Mean over non-padding tokens: (B, L, D) to (B, D)."""
    mask = pad_mask.to(hidden.dtype).unsqueeze(-1)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
