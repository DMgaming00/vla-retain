"""A real SmolVLAPolicy at a tiny scale, built offline.

We patch only the two Hub lookups in LeRobot's SmolVLMWithExpertModel, so the tests run LeRobot's own forward,
sampling, and freezing code on a model small enough for CPU.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import torch
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from transformers import SmolVLMConfig

IMG_KEY = "observation.images.image"
IMG_KEY2 = "observation.images.image2"
STATE_DIM = 8
ACTION_DIM = 7
CHUNK = 6
NUM_LAYERS = 6
IMG_SIZE = 32


def tiny_vlm_config() -> SmolVLMConfig:
    return SmolVLMConfig(
        text_config={
            "model_type": "llama",
            "vocab_size": 64,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": NUM_LAYERS,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 16,
        },
        vision_config={
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "image_size": IMG_SIZE,
            "patch_size": 8,
        },
        scale_factor=2,
    )


def tiny_policy_config(**overrides) -> SmolVLAConfig:
    kwargs = dict(
        device="cpu",
        chunk_size=CHUNK,
        n_action_steps=CHUNK,
        num_steps=3,
        max_state_dim=16,
        max_action_dim=16,
        resize_imgs_with_padding=(IMG_SIZE, IMG_SIZE),
        num_vlm_layers=NUM_LAYERS,
        tokenizer_max_length=4,
        input_features={
            IMG_KEY: PolicyFeature(type=FeatureType.VISUAL, shape=(3, IMG_SIZE, IMG_SIZE)),
            IMG_KEY2: PolicyFeature(type=FeatureType.VISUAL, shape=(3, IMG_SIZE, IMG_SIZE)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))},
    )
    kwargs.update(overrides)
    return SmolVLAConfig(**kwargs)


def make_tiny_policy(seed: int = 0, **overrides) -> SmolVLAPolicy:
    torch.manual_seed(seed)
    tokenizer = SimpleNamespace(fake_image_token_id=1, global_image_token_id=2)
    path = "lerobot.policies.smolvla.smolvlm_with_expert"
    with (
        patch(f"{path}.AutoConfig.from_pretrained", return_value=tiny_vlm_config()),
        patch(f"{path}.AutoProcessor.from_pretrained", return_value=SimpleNamespace(tokenizer=tokenizer)),
    ):
        policy = SmolVLAPolicy(tiny_policy_config(**overrides))
    return policy.to(dtype=torch.float32)


def make_batch(n: int = 4, seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {
        IMG_KEY: torch.rand(n, 3, IMG_SIZE, IMG_SIZE, generator=g),
        IMG_KEY2: torch.rand(n, 3, IMG_SIZE, IMG_SIZE, generator=g),
        OBS_STATE: torch.randn(n, STATE_DIM, generator=g),
        OBS_LANGUAGE_TOKENS: torch.randint(3, 64, (n, 4), generator=g),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(n, 4, dtype=torch.bool),
        ACTION: torch.randn(n, CHUNK, ACTION_DIM, generator=g),
        "action_is_pad": torch.zeros(n, CHUNK, dtype=torch.bool),
    }
