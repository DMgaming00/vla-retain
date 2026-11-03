import torch
from peft import LoraConfig, get_peft_model_state_dict, inject_adapter_in_model
import pytest

from tests.tiny import NUM_LAYERS, make_tiny_policy
from vlaretain.methods.lora import lora_target_regex
from vlaretain.methods.merge import (
    AlphaSpec,
    global_alpha_grid,
    layer_alpha_grid,
    lora_merged_state_dict,
    merge_state_dicts,
    module_alpha_grid,
)
from vlaretain.models.modules import module_group


def _pair():
    s1, ft = make_tiny_policy(seed=0).state_dict(), make_tiny_policy(seed=1).state_dict()
    return {k: v.clone() for k, v in s1.items()}, {k: v.clone() for k, v in ft.items()}


def _params(sd):
    return {k: v for k, v in sd.items() if torch.is_floating_point(v) and "rotary_emb" not in k}


def test_alpha_zero_gives_s1_exactly():
    s1, ft = _pair()
    merged = merge_state_dicts(s1, ft, AlphaSpec.parse("global=0"), NUM_LAYERS)
    assert all(torch.equal(merged[k], s1[k]) for k in s1)


def test_alpha_one_gives_ft_exactly():
    s1, ft = _pair()
    merged = merge_state_dicts(s1, ft, AlphaSpec.parse("global=1"), NUM_LAYERS)
    assert all(torch.equal(merged[k], ft[k]) for k in _params(s1))


def test_midpoint_is_linear():
    s1, ft = _pair()
    merged = merge_state_dicts(s1, ft, AlphaSpec.parse("global=0.3"), NUM_LAYERS)
    for key in _params(s1):
        assert torch.allclose(merged[key], 0.7 * s1[key] + 0.3 * ft[key], atol=1e-6)


def test_per_module_alpha_routes_to_the_right_groups():
    s1, ft = _pair()
    merged = merge_state_dicts(s1, ft, AlphaSpec.parse("lm=0.25,expert=1"), NUM_LAYERS)
    for key in _params(s1):
        group = module_group(key, NUM_LAYERS)
        if group == "expert":
            assert torch.equal(merged[key], ft[key])
        elif group == "vision":
            assert torch.equal(merged[key], s1[key])
        else:
            assert torch.allclose(merged[key], 0.75 * s1[key] + 0.25 * ft[key], atol=1e-6)


def test_layer_merge_uses_group_alphas():
    s1, ft = _pair()
    merged = merge_state_dicts(s1, ft, AlphaSpec.parse("lm_early=0,lm_mid=0.5,lm_late=1,expert=0"), NUM_LAYERS)
    for key in _params(s1):
        group = module_group(key, NUM_LAYERS)
        if group == "lm_early":
            assert torch.equal(merged[key], s1[key])
        if group == "lm_late":
            assert torch.equal(merged[key], ft[key])


def test_merged_policy_runs_and_nothing_touches_disk(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    s1, ft = _pair()
    policy = make_tiny_policy()
    policy.load_state_dict(merge_state_dicts(s1, ft, AlphaSpec.parse("global=0.5"), NUM_LAYERS))
    assert list(tmp_path.iterdir()) == []


def test_alpha_spec_validation_and_canonical_form():
    assert AlphaSpec.parse("expert=0.5,lm=0.25").canonical == "expert=0.50,lm=0.25"
    assert AlphaSpec.parse("global=0.3").canonical == "global=0.30"
    with pytest.raises(ValueError):
        AlphaSpec.parse("global=1.5")
    with pytest.raises(ValueError):
        AlphaSpec.parse("lm=0.5")
    with pytest.raises(ValueError):
        AlphaSpec.parse("lm=0.5,lm_mid=0.1,expert=1")


def test_grids_match_the_protocol():
    assert len(global_alpha_grid()) == 11 and global_alpha_grid()[0] == "global=0.00"
    assert len(module_alpha_grid()) == 16
    assert len(layer_alpha_grid()) == 27


def test_merged_lora_weights_match_peft():
    """Our baked weights must equal what PEFT produces when it merges the adapter into the model itself."""
    policy = make_tiny_policy(seed=2)
    base = {k: v.clone() for k, v in policy.state_dict().items()}
    config = LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=lora_target_regex())
    inject_adapter_in_model(config, policy)
    for name, param in policy.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(param, std=0.05)
    adapter = {k: v.clone() for k, v in get_peft_model_state_dict(policy).items()}
    assert adapter

    ours = lora_merged_state_dict(base, adapter, scale=8 / 4)

    for module in policy.modules():
        if hasattr(module, "merge") and hasattr(module, "lora_A"):
            module.merge()
    theirs = policy.state_dict()
    changed = 0
    for key, value in ours.items():
        peft_key = key.replace(".weight", ".base_layer.weight") if f"{key[:-7]}.base_layer.weight" in theirs else key
        assert torch.allclose(value, theirs[peft_key], atol=1e-6), key
        changed += int(not torch.equal(value, base[key]))
    assert changed > 0
